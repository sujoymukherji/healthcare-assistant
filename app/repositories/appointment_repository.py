from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from app.repositories.local_database import get_local_database_repository
from app.schemas.domain import ActorType, Appointment, AppointmentStatus, Doctor


class AppointmentRepository:
    """SQLite-backed appointment board for patient, attendant, and doctor workflows."""
    SLOT_DURATION_MINUTES = 60
    SEED_BUSINESS_DATES = 7

    def __init__(self) -> None:
        self._database = get_local_database_repository()
        self._seed_defaults()
        self._migrate_legacy_doctor_ids()
        self._seed_open_slots_from(date.today())

    def _seed_defaults(self) -> None:
        if self._database.count_appointments() > 0:
            return
        if not self._active_doctors():
            return
        for appointment in self._default_appointments():
            self._database.upsert_appointment(appointment)
        self._seed_open_slots_from(date.today())

    def reset_demo_schedule(self) -> list[Appointment]:
        self._database.clear_appointments()
        seeded = self._default_appointments()
        for appointment in seeded:
            self._database.upsert_appointment(appointment)
        self._seed_open_slots_from(date.today())
        return self.list_doctor_schedule()

    def _doctor_lookup(self) -> dict[str, Doctor]:
        return {doctor.doctor_id: doctor for doctor in self._database.list_doctors()}

    def _normalize_doctor_query(self, value: str) -> str:
        normalized = value.strip().lower().replace('.', ' ').replace('-', ' ')
        normalized = ' '.join(normalized.split())
        normalized = normalized.removeprefix('dr ')
        normalized = normalized.removeprefix('doctor ')
        return normalized.strip()

    def _normalize_specialty_query(self, value: str) -> str:
        normalized = self._normalize_doctor_query(value)
        for token in ('department', 'dept', 'specialty', 'speciality', 'doctor', 'physician', 'surgeon', 'specialist', 'clinic'):
            normalized = re.sub(rf'\b{re.escape(token)}\b', ' ', normalized)
        normalized = ' '.join(normalized.split()).strip()
        alias_map = {
            'orthopaedic': 'orthopedic',
            'orthopaedics': 'orthopedic',
            'ortho': 'orthopedic',
            'orthopedic surgeon': 'orthopedic',
            'orthopaedic surgeon': 'orthopedic',
            'general physician': 'general medicine',
            'general practitioner': 'general medicine',
            'general med': 'general medicine',
            'gen med': 'general medicine',
            'medicine': 'general medicine',
            'pulmonologist': 'pulmonology',
            'pulmonary': 'pulmonology',
            'pulm': 'pulmonology',
            'cardiologist': 'cardiology',
            'cardiac': 'cardiology',
            'cardio': 'cardiology',
        }
        return alias_map.get(normalized, normalized)

    def _specialty_keys_for_doctor(self, specialty: str) -> set[str]:
        normalized = self._normalize_specialty_query(specialty)
        keys = {normalized, self._normalize_doctor_query(specialty)}
        specialty_aliases = {
            'orthopedic': {'orthopedic', 'orthopaedic', 'orthopedic surgeon', 'orthopaedic surgeon', 'ortho'},
            'general medicine': {'general medicine', 'general physician', 'general practitioner', 'general med', 'gen med', 'medicine'},
            'pulmonology': {'pulmonology', 'pulmonologist', 'pulmonary', 'pulm'},
            'cardiology': {'cardiology', 'cardiologist', 'cardiac', 'cardio'},
        }
        keys.update(specialty_aliases.get(normalized, set()))
        return {self._normalize_specialty_query(item) for item in keys if item}

    def _active_doctors(self) -> list[Doctor]:
        doctors = [doctor for doctor in self._database.list_doctors() if doctor.active]
        return sorted(doctors, key=lambda doctor: doctor.full_name.lower())

    def _default_doctor_id(self, specialty_hint: str | None = None) -> str:
        doctors = self._active_doctors()
        if not doctors:
            return 'doc_unassigned'
        if specialty_hint:
            normalized_hint = specialty_hint.strip().lower().replace('_', ' ')
            for doctor in doctors:
                specialty = doctor.specialty.lower().replace('_', ' ')
                if (
                    normalized_hint in specialty
                    or specialty in normalized_hint
                    or specialty[:6] in normalized_hint
                    or normalized_hint[:6] in specialty
                ):
                    return doctor.doctor_id
        return doctors[0].doctor_id

    def _legacy_doctor_map(self) -> dict[str, str]:
        return {
            'doc_gen_001': self._default_doctor_id('general medicine'),
            'doc_pulm_001': self._default_doctor_id('pulmonology'),
        }

    def _migrate_legacy_doctor_ids(self) -> None:
        legacy_map = self._legacy_doctor_map()
        if not self._active_doctors():
            return
        for appointment in self._database.list_appointments():
            replacement = legacy_map.get(appointment.doctor_id)
            if replacement and replacement != appointment.doctor_id:
                self._database.upsert_appointment(appointment.model_copy(update={'doctor_id': replacement}))

    def _default_appointments(self) -> list[Appointment]:
        james_id = self._default_doctor_id('general medicine')
        return [
            Appointment(
                appointment_id='appt_001',
                patient_id='pat_david_thompson',
                doctor_id=james_id,
                specialty='general_medicine',
                status=AppointmentStatus.BOOKED,
                appointment_date=date(2026, 4, 10),
                appointment_time=time(10, 30),
                location_id='clinic_main',
                booked_by_actor=ActorType.PATIENT,
                booking_reason='Diabetes follow-up',
                created_at=datetime(2026, 4, 1, 9, 0),
            ),
            Appointment(
                appointment_id='appt_002',
                patient_id='pat_anjali_mehra',
                doctor_id=james_id,
                specialty='general_medicine',
                status=AppointmentStatus.BOOKED,
                appointment_date=date(2026, 4, 10),
                appointment_time=time(11, 0),
                location_id='clinic_main',
                booked_by_actor=ActorType.PATIENT,
                booking_reason='Cough and mild fever',
                created_at=datetime(2026, 4, 2, 10, 0),
            ),
        ]

    def _business_dates_from(self, anchor_date: date) -> list[date]:
        dates: list[date] = []
        current = anchor_date
        while len(dates) < self.SEED_BUSINESS_DATES:
            if current.weekday() < 6:
                dates.append(current)
            current += timedelta(days=1)
        return dates

    def _slots_for_window(self, doctor: Doctor, slot_date: date) -> list[Appointment]:
        slots: list[Appointment] = []
        for window in doctor.availability:
            if slot_date.weekday() not in window.days_of_week:
                continue
            cursor = datetime.combine(slot_date, window.start_time)
            window_end = datetime.combine(slot_date, window.end_time)
            while cursor + timedelta(minutes=self.SLOT_DURATION_MINUTES) <= window_end:
                appointment_time = cursor.time()
                slot_id = f"slot_{doctor.doctor_id}_{slot_date.strftime('%Y%m%d')}_{appointment_time.strftime('%H%M')}"
                slots.append(
                    Appointment(
                        appointment_id=slot_id,
                        patient_id='',
                        doctor_id=doctor.doctor_id,
                        specialty=doctor.specialty.lower().replace(' ', '_'),
                        status=AppointmentStatus.AVAILABLE,
                        appointment_date=slot_date,
                        appointment_time=appointment_time,
                        location_id=doctor.location_id,
                        booked_by_actor=ActorType.IT_ADMIN,
                        booking_reason='Availability seed slot',
                        created_at=datetime.now(),
                    )
                )
                cursor += timedelta(minutes=self.SLOT_DURATION_MINUTES)
        return slots

    def _has_any_slots_for_date(self, target_date: date) -> bool:
        return any(appointment.appointment_date == target_date for appointment in self._database.list_appointments())

    def _seed_open_slots_from(self, anchor_date: date) -> None:
        doctors = self._active_doctors()
        if not doctors:
            return
        existing_ids = {appointment.appointment_id for appointment in self._database.list_appointments()}
        for slot_date in self._business_dates_from(anchor_date):
            if self._has_any_slots_for_date(slot_date):
                continue
            for doctor in doctors:
                for slot in self._slots_for_window(doctor, slot_date):
                    if slot.appointment_id in existing_ids:
                        continue
                    self._database.upsert_appointment(slot)
                    existing_ids.add(slot.appointment_id)

    def _all_appointments(self) -> list[Appointment]:
        if self._database.count_appointments() == 0:
            self._seed_defaults()
        self._migrate_legacy_doctor_ids()
        return self._database.list_appointments()

    def list_open_appointments(self) -> list[Appointment]:
        today = date.today()
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.status == AppointmentStatus.AVAILABLE and appt.appointment_date >= today
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_open_appointments_for_range(
        self,
        start_date: date,
        end_date: date,
        *,
        doctor_query: str | None = None,
    ) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.status == AppointmentStatus.AVAILABLE and start_date <= appt.appointment_date <= end_date
        ]
        if doctor_query:
            doctor_id = self.resolve_doctor_id(doctor_query)
            if doctor_id is None:
                return []
            appointments = [appt for appt in appointments if appt.doctor_id == doctor_id]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_booked_appointments(self) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_doctor_schedule(self) -> list[Appointment]:
        return sorted(self._all_appointments(), key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_appointments_for_date(self, appointment_date: date) -> list[Appointment]:
        appointments = [appt for appt in self._all_appointments() if appt.appointment_date == appointment_date]
        return sorted(appointments, key=lambda appt: (appt.appointment_time, appt.appointment_id))

    def list_appointments_for_range(self, start_date: date, end_date: date) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if start_date <= appt.appointment_date <= end_date
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_booked_appointments_for_date(self, appointment_date: date) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.appointment_date == appointment_date and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_time, appt.appointment_id))

    def list_booked_appointments_for_range(self, start_date: date, end_date: date) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if start_date <= appt.appointment_date <= end_date and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def get_doctor_display_name(self, doctor_id: str) -> str:
        doctor_id = self._legacy_doctor_map().get(doctor_id, doctor_id)
        doctor = self._doctor_lookup().get(doctor_id)
        return doctor.full_name if doctor is not None else doctor_id

    def get_doctor_by_id(self, doctor_id: str) -> Doctor | None:
        return self._doctor_lookup().get(doctor_id)

    def resolve_doctor_id(self, doctor_query: str | None) -> str | None:
        if not doctor_query:
            return None
        normalized = self._normalize_doctor_query(doctor_query)
        normalized_specialty = self._normalize_specialty_query(doctor_query)
        specialty_aliases = {
            'orthopaedic': 'orthopedic',
            'orthopedics': 'orthopedic',
            'general physician': 'general medicine',
            'general practitioner': 'general medicine',
            'pulmonologist': 'pulmonology',
            'cardiologist': 'cardiology',
            'dermatologist': 'dermatology',
            'neurologist': 'neurology',
            'ophthalmologist': 'ophthalmology',
        }
        normalized = specialty_aliases.get(normalized, normalized)
        legacy_aliases = {
            'doctor a': self._legacy_doctor_map().get('doc_gen_001'),
            'doctor b': self._legacy_doctor_map().get('doc_pulm_001'),
        }
        if normalized in legacy_aliases and legacy_aliases[normalized]:
            return legacy_aliases[normalized]
        for doctor in self._active_doctors():
            specialty_keys = self._specialty_keys_for_doctor(doctor.specialty)
            accepted = {
                self._normalize_doctor_query(doctor.doctor_id),
                self._normalize_doctor_query(doctor.full_name),
                self._normalize_doctor_query(doctor.specialty),
                self._normalize_doctor_query(doctor.specialty.replace('_', ' ')),
            }
            if doctor.phone:
                accepted.add(''.join(ch for ch in doctor.phone if ch.isdigit()))
            if doctor.email:
                accepted.add(doctor.email.lower())
            if normalized in accepted or normalized_specialty in specialty_keys:
                return doctor.doctor_id
            if any(
                normalized_specialty in key or key in normalized_specialty
                for key in specialty_keys
                if key
            ):
                return doctor.doctor_id
        return None

    def list_patient_appointments(self, patient_id: str) -> list[Appointment]:
        appointments = [appt for appt in self._all_appointments() if appt.patient_id == patient_id]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id), reverse=True)

    def list_cancelled_patient_appointments(self, patient_id: str) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.patient_id == patient_id and appt.status == AppointmentStatus.CANCELLED
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id), reverse=True)

    def list_current_patient_appointments(self, patient_id: str) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.patient_id == patient_id and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id), reverse=True)

    def list_open_appointments_for_date(self, appointment_date: date) -> list[Appointment]:
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.status == AppointmentStatus.AVAILABLE and appt.appointment_date == appointment_date
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_time, appt.appointment_id))

    def get_appointment_by_id(self, appointment_id: str) -> Appointment | None:
        return next((appt for appt in self._all_appointments() if appt.appointment_id == appointment_id), None)

    def ensure_open_slots_for_range(self, start_date: date, end_date: date) -> None:
        current = start_date
        while current <= end_date:
            if current.weekday() < 6 and not self._has_any_slots_for_date(current):
                self._seed_open_slots_from(current)
            current += timedelta(days=1)

    def list_doctor_appointments_for_date(self, doctor_query: str, appointment_date: date) -> list[Appointment]:
        doctor_id = self.resolve_doctor_id(doctor_query)
        if doctor_id is None:
            return []
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.doctor_id == doctor_id
            and appt.appointment_date == appointment_date
            and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_time, appt.appointment_id))

    def list_doctor_appointments(self, doctor_query: str) -> list[Appointment]:
        doctor_id = self.resolve_doctor_id(doctor_query)
        if doctor_id is None:
            return []
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.doctor_id == doctor_id and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def list_doctor_appointments_for_range(self, doctor_query: str, start_date: date, end_date: date) -> list[Appointment]:
        doctor_id = self.resolve_doctor_id(doctor_query)
        if doctor_id is None:
            return []
        appointments = [
            appt
            for appt in self._all_appointments()
            if appt.doctor_id == doctor_id
            and start_date <= appt.appointment_date <= end_date
            and appt.status in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}
        ]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def find_matching_open_slots(
        self,
        *,
        target_date: date | None = None,
        end_date: date | None = None,
        doctor_query: str | None = None,
        target_time: time | None = None,
    ) -> list[Appointment]:
        appointments = self.list_open_appointments()
        if target_date is not None and end_date is not None:
            appointments = [appt for appt in appointments if target_date <= appt.appointment_date <= end_date]
        elif target_date is not None:
            appointments = [appt for appt in appointments if appt.appointment_date == target_date]
        if doctor_query:
            doctor_id = self.resolve_doctor_id(doctor_query)
            if doctor_id is None:
                return []
            appointments = [appt for appt in appointments if appt.doctor_id == doctor_id]
        if target_time is not None:
            appointments = [appt for appt in appointments if appt.appointment_time == target_time]
        return sorted(appointments, key=lambda appt: (appt.appointment_date, appt.appointment_time, appt.appointment_id))

    def book_appointment(
        self,
        appointment_id: str,
        *,
        patient_id: str,
        booking_reason: str,
        booked_by_actor: ActorType,
    ) -> Appointment | None:
        appointment = self.get_appointment_by_id(appointment_id)
        if appointment is None or appointment.status != AppointmentStatus.AVAILABLE:
            return None
        booked = appointment.model_copy(
            update={
                'patient_id': patient_id,
                'status': AppointmentStatus.BOOKED,
                'booked_by_actor': booked_by_actor,
                'booking_reason': booking_reason,
            }
        )
        self._database.upsert_appointment(booked)
        return booked

    def schedule_next_available_for_patient(
        self,
        *,
        patient_id: str,
        booking_reason: str,
        booked_by_actor: ActorType,
    ) -> Appointment | None:
        open_slots = self.list_open_appointments()
        if not open_slots:
            return None
        return self.book_appointment(
            open_slots[0].appointment_id,
            patient_id=patient_id,
            booking_reason=booking_reason,
            booked_by_actor=booked_by_actor,
        )

    def schedule_matching_available_for_patient(
        self,
        *,
        patient_id: str,
        booking_reason: str,
        booked_by_actor: ActorType,
        target_date: date | None = None,
        target_end_date: date | None = None,
        doctor_query: str | None = None,
        target_time: time | None = None,
    ) -> Appointment | None:
        matches = self.find_matching_open_slots(
            target_date=target_date,
            end_date=target_end_date,
            doctor_query=doctor_query,
            target_time=target_time,
        )
        if target_date is not None and target_end_date is not None and not matches:
            self.ensure_open_slots_for_range(target_date, target_end_date)
            matches = self.find_matching_open_slots(
                target_date=target_date,
                end_date=target_end_date,
                doctor_query=doctor_query,
                target_time=target_time,
            )
        elif target_date is not None and not matches and not self._has_any_slots_for_date(target_date):
            self._seed_open_slots_from(target_date)
            matches = self.find_matching_open_slots(
                target_date=target_date,
                doctor_query=doctor_query,
                target_time=target_time,
            )
        if not matches:
            return None
        return self.book_appointment(
            matches[0].appointment_id,
            patient_id=patient_id,
            booking_reason=booking_reason,
            booked_by_actor=booked_by_actor,
        )

    def reschedule_appointment(
        self,
        appointment_id: str,
        *,
        patient_id: str,
        new_date: date,
        booked_by_actor: ActorType,
    ) -> Appointment | None:
        appointment = self.get_appointment_by_id(appointment_id)
        if appointment is None or appointment.patient_id != patient_id or appointment.status not in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}:
            return None
        updated = appointment.model_copy(
            update={
                'appointment_date': new_date,
                'booked_by_actor': booked_by_actor,
            }
        )
        self._database.upsert_appointment(updated)
        return updated

    def rebook_appointment(
        self,
        appointment_id: str,
        *,
        patient_id: str,
        booking_reason: str,
        booked_by_actor: ActorType,
        target_date: date,
        doctor_query: str | None = None,
        target_time: time | None = None,
    ) -> tuple[Appointment | None, Appointment | None]:
        current = self.get_appointment_by_id(appointment_id)
        if current is None or current.patient_id != patient_id or current.status not in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}:
            return None, None
        reopened = self.cancel_appointment(appointment_id, patient_id=patient_id, booked_by_actor=booked_by_actor)
        if reopened is None:
            return None, None
        booked = self.schedule_matching_available_for_patient(
            patient_id=patient_id,
            booking_reason=booking_reason,
            booked_by_actor=booked_by_actor,
            target_date=target_date,
            doctor_query=doctor_query,
            target_time=target_time,
        )
        return reopened, booked

    def rebook_appointment_to_matching_available(
        self,
        appointment_id: str,
        *,
        patient_id: str,
        booking_reason: str,
        booked_by_actor: ActorType,
        target_date: date,
        target_end_date: date | None = None,
        doctor_query: str | None = None,
        target_time: time | None = None,
    ) -> tuple[Appointment | None, Appointment | None]:
        current = self.get_appointment_by_id(appointment_id)
        if current is None or current.patient_id != patient_id or current.status not in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}:
            return None, None
        reopened = self.cancel_appointment(appointment_id, patient_id=patient_id, booked_by_actor=booked_by_actor)
        if reopened is None:
            return None, None
        booked = self.schedule_matching_available_for_patient(
            patient_id=patient_id,
            booking_reason=booking_reason,
            booked_by_actor=booked_by_actor,
            target_date=target_date,
            target_end_date=target_end_date,
            doctor_query=doctor_query,
            target_time=target_time,
        )
        return reopened, booked

    def cancel_appointment(
        self,
        appointment_id: str,
        *,
        patient_id: str,
        booked_by_actor: ActorType,
    ) -> Appointment | None:
        appointment = self.get_appointment_by_id(appointment_id)
        if appointment is None or appointment.patient_id != patient_id or appointment.status not in {AppointmentStatus.BOOKED, AppointmentStatus.HELD}:
            return None
        reopened = appointment.model_copy(
            update={
                'patient_id': '',
                'status': AppointmentStatus.AVAILABLE,
                'booked_by_actor': ActorType.IT_ADMIN,
                'booking_reason': 'Reopened after cancellation',
            }
        )
        self._database.upsert_appointment(reopened)
        return reopened

    def bulk_reschedule_doctor_appointments(
        self,
        *,
        doctor_query: str,
        source_date: date,
        new_date: date,
        booked_by_actor: ActorType,
    ) -> list[Appointment]:
        appointments = self.list_doctor_appointments_for_date(doctor_query, source_date)
        updated_appointments: list[Appointment] = []
        for appointment in appointments:
            updated = appointment.model_copy(
                update={
                    'appointment_date': new_date,
                    'booked_by_actor': booked_by_actor,
                }
            )
            self._database.upsert_appointment(updated)
            updated_appointments.append(updated)
        return updated_appointments


_APPOINTMENT_REPOSITORY = AppointmentRepository()


def get_appointment_repository() -> AppointmentRepository:
    return _APPOINTMENT_REPOSITORY
