from __future__ import annotations

from difflib import SequenceMatcher
from datetime import date, datetime
import re
import traceback
from uuid import uuid4

from app.repositories.local_database import get_local_database_repository
from app.schemas.domain import Diagnosis, Doctor, MedicalRecordEntry, Patient, RecordEntryType
from app.services.sample_data_loader import SampleDataLoader


class SampleDataRepository:
    """Repository backed by normalized sample data."""

    def __init__(self, loader: SampleDataLoader | None = None) -> None:
        self.loader = loader or SampleDataLoader()
        self.database = get_local_database_repository()
        bundle = self.loader.load_all()
        self._bootstrap_sample_entities(bundle)
        for doctor in bundle.doctors:
            self.database.upsert_doctor(Doctor.model_validate(doctor))
        self.patients = self.database.list_sample_patients()
        self.doctors = self.database.list_doctors()
        self.records = self.database.list_medical_records()
        self.diagnoses = [Diagnosis.model_validate(item) for item in bundle.diagnoses]
        self.workbook_patients = [Patient.model_validate(item) for item in bundle.workbook_patients]
        self.workbook_records = [MedicalRecordEntry.model_validate(item) for item in bundle.workbook_records]
        self.raw_spreadsheet_rows = bundle.raw_spreadsheet_rows
        self.raw_pdf_documents = bundle.raw_pdf_documents
        self._ensure_demo_phone_numbers()
        self._persist_workbook_entities()

    def _bootstrap_sample_entities(self, bundle) -> None:
        if (
            self.database.count_sample_patients() > 0
            and self.database.count_medical_records() > 0
            and self.database.count_doctors() > 0
        ):
            return
        self.database.clear_sample_snapshot()
        for doctor in bundle.doctors:
            self.database.upsert_doctor(Doctor.model_validate(doctor))
        for patient in bundle.patients:
            self.database.upsert_patient(Patient.model_validate(patient))
        for record in bundle.records:
            self.database.upsert_record(MedicalRecordEntry.model_validate(record))

    def _normalize_phone(self, value: str | None) -> str | None:
        if not value:
            return None
        digits = ''.join(ch for ch in value if ch.isdigit())
        return digits or None

    def _ensure_demo_phone_numbers(self) -> None:
        patient_index = {patient.patient_id: patient for patient in self.patients}
        for index, patient in enumerate(self.patients, start=1):
            normalized = self._normalize_phone(patient.phone)
            patient.phone = normalized or f'555000{index:04d}'
            self.database.upsert_patient(patient)
        for workbook_patient in self.workbook_patients:
            canonical = patient_index.get(workbook_patient.patient_id)
            if canonical is not None:
                workbook_patient.phone = canonical.phone

    def _persist_workbook_entities(self) -> None:
        self.database.clear_workbook_snapshot()
        for patient in self.workbook_patients:
            source_path = patient.source_refs[0].source_path if patient.source_refs else 'unknown'
            self.database.upsert_workbook_patient(patient, source_path)
        for record in self.workbook_records:
            self.database.upsert_workbook_record(record)

    def _split_name(self, full_name: str) -> tuple[str, str]:
        parts = [part for part in full_name.strip().split() if part]
        if not parts:
            return '', ''
        if len(parts) == 1:
            return parts[0], ''
        return parts[0], ' '.join(parts[1:])

    def _make_patient_id(self, first_name: str, last_name: str) -> str:
        def normalize(part: str) -> str:
            cleaned = re.sub(r'[^a-z0-9]+', '_', part.strip().lower())
            return cleaned.strip('_') or 'patient'

        prefix = f"{normalize(first_name)}_{normalize(last_name or 'patient')}"
        existing_ids = {patient.patient_id for patient in self.database.list_sample_patients()}
        sequence = 1
        candidate = f"{prefix}_{sequence:03d}"
        while candidate in existing_ids:
            sequence += 1
            candidate = f"{prefix}_{sequence:03d}"
        return candidate

    def get_patient_by_id(self, patient_id: str) -> Patient | None:
        return next((patient for patient in self.patients if patient.patient_id == patient_id), None)

    def get_patient_by_name(self, full_name: str) -> Patient | None:
        lowered = full_name.strip().lower()
        return next((patient for patient in self.patients if patient.full_name.lower() == lowered), None)

    def search_patients_by_name(self, query: str) -> list[Patient]:
        lowered = query.strip().lower()
        if not lowered:
            return []
        direct_matches = [patient for patient in self.patients if lowered in patient.full_name.lower()]
        if direct_matches:
            return direct_matches

        normalized_query = self._normalize_name_for_search(query)
        if not normalized_query:
            return []

        scored_matches: list[tuple[float, Patient]] = []
        for patient in self.patients:
            normalized_name = self._normalize_name_for_search(patient.full_name)
            if not normalized_name:
                continue
            score = SequenceMatcher(None, normalized_query, normalized_name).ratio()
            if score >= 0.82:
                scored_matches.append((score, patient))

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        return [patient for _, patient in scored_matches]

    def list_patients(self) -> list[Patient]:
        return sorted(self.patients, key=lambda patient: patient.full_name.lower())

    def list_doctors(self) -> list[Doctor]:
        return sorted(self.doctors, key=lambda doctor: doctor.full_name.lower())

    def get_doctor_by_id(self, doctor_id: str) -> Doctor | None:
        return next((doctor for doctor in self.doctors if doctor.doctor_id == doctor_id), None)

    def search_doctors_by_name(self, query: str) -> list[Doctor]:
        lowered = query.strip().lower()
        if not lowered:
            return []
        return [doctor for doctor in self.doctors if lowered in doctor.full_name.lower()]

    def _normalize_name_for_search(self, value: str) -> str:
        lowered = value.strip().lower()
        lowered = re.sub(r'[^a-z0-9\s]+', ' ', lowered)
        return ' '.join(lowered.split())

    def get_doctor_by_phone(self, phone: str) -> Doctor | None:
        normalized = self._normalize_phone(phone)
        if not normalized:
            return None
        return next((doctor for doctor in self.doctors if self._normalize_phone(doctor.phone) == normalized), None)

    def register_doctor(
        self,
        *,
        full_name: str,
        specialty: str,
        phone: str | None = None,
        email: str | None = None,
        gender: str | None = None,
        clinic_name: str = 'Apollo Health',
        location_id: str = 'clinic_main',
    ) -> Doctor:
        base_slug = re.sub(r'[^a-z0-9]+', '_', full_name.strip().lower()).strip('_') or 'doctor'
        existing_ids = {doctor.doctor_id for doctor in self.doctors}
        candidate = f'doc_{base_slug}'
        sequence = 1
        while candidate in existing_ids:
            sequence += 1
            candidate = f'doc_{base_slug}_{sequence:03d}'
        doctor = Doctor(
            doctor_id=candidate,
            full_name=full_name.strip(),
            specialty=specialty.strip() or 'General Medicine',
            clinic_name=clinic_name,
            location_id=location_id,
            phone=self._normalize_phone(phone) if phone else None,
            email=email.strip() if email else None,
            gender=gender.strip() if gender else None,
            active=True,
        )
        self.doctors.append(doctor)
        self.database.upsert_doctor(doctor)
        return doctor

    def update_doctor_details(
        self,
        doctor_id: str,
        *,
        full_name: str | None = None,
        specialty: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        gender: str | None = None,
    ) -> Doctor | None:
        doctor = self.get_doctor_by_id(doctor_id)
        if doctor is None:
            return None
        if full_name is not None and full_name.strip():
            doctor.full_name = full_name.strip()
        if specialty is not None and specialty.strip():
            doctor.specialty = specialty.strip()
        if phone is not None:
            doctor.phone = self._normalize_phone(phone) or doctor.phone
        if email is not None:
            doctor.email = email.strip() or None
        if gender is not None:
            doctor.gender = gender.strip() or None
        self.database.upsert_doctor(doctor)
        return doctor

    def get_patient_by_phone(self, phone: str) -> Patient | None:
        normalized = self._normalize_phone(phone)
        if not normalized:
            return None
        return next((patient for patient in self.patients if patient.phone == normalized), None)

    def register_patient(
        self,
        phone: str,
        first_name: str,
        last_name: str,
        address: str | None = None,
    ) -> Patient:
        normalized = self._normalize_phone(phone) or phone
        existing = self.get_patient_by_phone(normalized)
        if existing is not None:
            return existing
        clean_first = first_name.strip()
        clean_last = last_name.strip()
        full_name = f'{clean_first} {clean_last}'.strip()
        patient = Patient(
            patient_id=self._make_patient_id(clean_first, clean_last),
            full_name=full_name or 'New Patient',
            phone=normalized,
            address=address.strip() if address else None,
        )
        self.patients.append(patient)
        self.database.upsert_patient(patient)
        return patient

    def get_records_for_patient(self, patient_id: str) -> list[MedicalRecordEntry]:
        records = [record for record in self.records if record.patient_id == patient_id]
        return sorted(records, key=lambda record: (record.visit_date or date.min, record.record_id), reverse=True)

    def get_diagnoses_for_patient(self, patient_id: str) -> list[Diagnosis]:
        return [diagnosis for diagnosis in self.diagnoses if diagnosis.patient_id == patient_id]

    def get_recent_record_for_patient(self, patient_id: str) -> MedicalRecordEntry | None:
        records = self.get_records_for_patient(patient_id)
        return records[0] if records else None

    def update_patient_details(
        self,
        patient_id: str,
        *,
        first_name: str | None = None,
        last_name: str | None = None,
        address: str | None = None,
        phone: str | None = None,
    ) -> Patient | None:
        patient = self.get_patient_by_id(patient_id)
        if patient is None:
            return None
        current_first, current_last = self._split_name(patient.full_name)
        next_first = first_name.strip() if first_name is not None else current_first
        next_last = last_name.strip() if last_name is not None else current_last
        patient.full_name = f'{next_first} {next_last}'.strip() or patient.full_name
        if address is not None:
            patient.address = address.strip() or None
        if phone is not None:
            patient.phone = self._normalize_phone(phone) or patient.phone
        self.database.upsert_patient(patient)
        return patient

    def delete_patient(self, patient_id: str) -> None:
        self.patients = [patient for patient in self.patients if patient.patient_id != patient_id]
        self.records = [record for record in self.records if record.patient_id != patient_id]
        self.diagnoses = [diagnosis for diagnosis in self.diagnoses if diagnosis.patient_id != patient_id]
        self.database.delete_sample_patient(patient_id)

    def add_clinical_note(
        self,
        patient_id: str,
        note_text: str | None,
        treatment_text: str | None,
        *,
        doctor_id: str = 'doc_james_c',
    ) -> MedicalRecordEntry:
        record_id = f'rec_manual_{len(self.records) + 1:03d}'
        title = 'Doctor Follow-up Note'
        structured_fields = {'writeback': True}
        if treatment_text:
            structured_fields['prescribed_treatment'] = treatment_text
        entry = MedicalRecordEntry(
            record_id=record_id,
            patient_id=patient_id,
            entry_type=RecordEntryType.ADMIN_NOTE,
            visit_date=date.today(),
            title=title,
            subjective=note_text,
            assessment=None,
            plan=treatment_text,
            doctor_id=doctor_id,
            source_type='manual_doctor_entry',
            source_path=f'in-memory://records/{record_id}',
            structured_fields=structured_fields,
        )
        self.records.append(entry)
        self.database.upsert_record(entry)
        return entry

    def apply_patient_history_update(
        self,
        patient_id: str,
        *,
        title: str,
        subjective: str | None,
        objective: str | None,
        assessment: str | None,
        plan: str | None,
        visit_date: date | None,
        source_type: str,
        source_path: str,
        doctor_id: str | None,
        primary_conditions: list[str] | None = None,
        chronic_conditions: list[str] | None = None,
        allergies: list[str] | None = None,
        cleared_conditions: list[str] | None = None,
        latest_visit_summary: str | None = None,
        summary_text: str | None = None,
    ) -> tuple[Patient | None, MedicalRecordEntry | None, dict[str, object] | None]:
        patient = self.get_patient_by_id(patient_id)
        if patient is None:
            return None, None, None

        cleared_lookup = {item.strip().lower() for item in (cleared_conditions or []) if item and item.strip()}
        if cleared_lookup:
            patient.primary_conditions = [
                condition for condition in patient.primary_conditions if condition.strip().lower() not in cleared_lookup
            ]
            patient.chronic_conditions = [
                condition for condition in patient.chronic_conditions if condition.strip().lower() not in cleared_lookup
            ]

        for condition in primary_conditions or []:
            if condition and condition not in patient.primary_conditions:
                patient.primary_conditions.append(condition)
        for condition in chronic_conditions or []:
            if condition and condition not in patient.chronic_conditions:
                patient.chronic_conditions.append(condition)
            if condition and condition not in patient.primary_conditions:
                patient.primary_conditions.append(condition)
        for allergy in allergies or []:
            if allergy and allergy not in patient.allergies:
                patient.allergies.append(allergy)

        self.database.upsert_patient(patient)

        record_id = f'rec_manual_{len(self.records) + 1:03d}'
        entry_type = RecordEntryType.UPLOADED_DOCUMENT if source_type == 'uploaded_pdf' else RecordEntryType.VISIT_NOTE
        record = MedicalRecordEntry(
            record_id=record_id,
            patient_id=patient_id,
            entry_type=entry_type,
            visit_date=visit_date or date.today(),
            title=title,
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            doctor_id=doctor_id,
            source_type=source_type,
            source_path=source_path,
            structured_fields={
                'writeback': True,
                'visit_summary': latest_visit_summary,
                'cleared_conditions': cleared_conditions or [],
                'primary_conditions': primary_conditions or [],
                'chronic_conditions': chronic_conditions or [],
                'allergies': allergies or [],
            },
        )
        self.records.append(record)
        self.database.upsert_record(record)

        self._upsert_history_summary_record(
            patient=patient,
            latest_visit_summary=latest_visit_summary,
            summary_text=summary_text,
        )
        return patient, record, self.build_patient_history_summary(patient_id)

    def build_patient_visit_history(self, patient_id: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for record in self.get_records_for_patient(patient_id):
            if record.entry_type == RecordEntryType.HISTORY_SUMMARY and not record.visit_date:
                continue
            summary = str(record.structured_fields.get('visit_summary') or record.plan or '').strip()
            rows.append(
                {
                    'Date': record.visit_date.isoformat() if record.visit_date else 'unknown',
                    'Doctor': record.doctor_id or 'unknown',
                    'Diagnosis': (record.assessment or record.title or 'n/a').strip(),
                    'Treatment Summary': summary or 'No treatment summary available',
                }
            )
        return rows

    def build_patient_history_summary(self, patient_id: str) -> dict[str, object]:
        patient = self.get_patient_by_id(patient_id)
        records = self.get_records_for_patient(patient_id)
        summary_record = next(
            (
                record
                for record in records
                if record.entry_type == RecordEntryType.HISTORY_SUMMARY and record.source_type in {'derived_summary', 'manual_summary', 'workbook'}
            ),
            None,
        )
        latest_record = next(
            (
                record
                for record in records
                if record.entry_type != RecordEntryType.HISTORY_SUMMARY
                and (record.assessment or record.plan or record.structured_fields.get('visit_summary'))
            ),
            None,
        )
        latest_summary = ''
        if latest_record is not None:
            latest_summary = str(
                latest_record.structured_fields.get('visit_summary') or latest_record.plan or latest_record.assessment or latest_record.title
            ).strip()
        summary_text = None
        if summary_record is not None:
            summary_text = str(
                summary_record.structured_fields.get('summary_text') or summary_record.subjective or ''
            ).strip() or None
            latest_summary = str(
                summary_record.structured_fields.get('latest_visit_summary') or latest_summary
            ).strip()
        return {
            'patient_name': patient.full_name if patient else '',
            'primary_conditions': patient.primary_conditions if patient else [],
            'chronic_conditions': patient.chronic_conditions if patient else [],
            'allergies': patient.allergies if patient else [],
            'record_count': len(records),
            'latest_visit_date': latest_record.visit_date.isoformat() if latest_record and latest_record.visit_date else None,
            'latest_visit_summary': latest_summary or None,
            'summary_text': summary_text,
        }

    def _upsert_history_summary_record(
        self,
        *,
        patient: Patient,
        latest_visit_summary: str | None,
        summary_text: str | None,
    ) -> None:
        record_id = f'summary_{patient.patient_id}'
        summary_record = next((record for record in self.records if record.record_id == record_id), None)
        if summary_record is None:
            summary_record = MedicalRecordEntry(
                record_id=record_id,
                patient_id=patient.patient_id,
                entry_type=RecordEntryType.HISTORY_SUMMARY,
                title='Updated Patient Summary',
                subjective=summary_text,
                source_type='derived_summary',
                source_path=f'in-memory://summaries/{patient.patient_id}',
                structured_fields={},
            )
            self.records.append(summary_record)
        summary_record.subjective = summary_text
        summary_record.structured_fields = {
            'summary_text': summary_text,
            'latest_visit_summary': latest_visit_summary,
            'primary_conditions': patient.primary_conditions,
            'chronic_conditions': patient.chronic_conditions,
            'allergies': patient.allergies,
        }
        self.database.upsert_record(summary_record)

    def log_patient_interaction(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str | None,
        workflow_type: str | None,
        context: dict[str, object] | None,
    ) -> None:
        self.database.log_interaction(
            interaction_id=f'interaction_{uuid4().hex[:12]}',
            session_id=session_id,
            actor='patient',
            workflow_type=workflow_type,
            user_message=user_message,
            assistant_message=assistant_message,
            context=context,
            created_at=datetime.now(),
        )

    def log_interaction(
        self,
        *,
        actor: str,
        session_id: str,
        user_message: str,
        assistant_message: str | None,
        workflow_type: str | None,
        context: dict[str, object] | None,
    ) -> None:
        self.database.log_interaction(
            interaction_id=f'interaction_{uuid4().hex[:12]}',
            session_id=session_id,
            actor=actor,
            workflow_type=workflow_type,
            user_message=user_message,
            assistant_message=assistant_message,
            context=context,
            created_at=datetime.now(),
        )

    def log_patient_planner_trace(
        self,
        *,
        session_id: str,
        user_message: str,
        context: dict[str, object] | None,
        planner_output: dict[str, object] | None,
        final_workflow_type: str | None,
    ) -> None:
        self.database.log_planner_trace(
            trace_id=f'planner_{uuid4().hex[:12]}',
            session_id=session_id,
            actor='patient',
            user_message=user_message,
            context=context,
            planner_output=planner_output,
            final_workflow_type=final_workflow_type,
            created_at=datetime.now(),
        )

    def log_planner_trace(
        self,
        *,
        actor: str,
        session_id: str,
        user_message: str,
        context: dict[str, object] | None,
        planner_output: dict[str, object] | None,
        final_workflow_type: str | None,
    ) -> None:
        self.database.log_planner_trace(
            trace_id=f'planner_{uuid4().hex[:12]}',
            session_id=session_id,
            actor=actor,
            user_message=user_message,
            context=context,
            planner_output=planner_output,
            final_workflow_type=final_workflow_type,
            created_at=datetime.now(),
        )

    def log_langsmith_run(
        self,
        *,
        session_id: str,
        actor: str,
        workflow_type: str | None,
        run_id: str,
        trace_url: str | None,
    ) -> None:
        self.database.log_langsmith_run(
            run_id=run_id,
            session_id=session_id,
            actor=actor,
            workflow_type=workflow_type,
            trace_url=trace_url,
            created_at=datetime.now(),
        )

    def log_system_error(
        self,
        *,
        session_id: str,
        actor: str,
        stage: str,
        error: Exception,
        retryable: bool = False,
    ) -> None:
        self.database.log_system_error(
            error_id=f'error_{uuid4().hex[:12]}',
            session_id=session_id,
            actor=actor,
            stage=stage,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            retryable=retryable,
            created_at=datetime.now(),
        )


_SAMPLE_REPOSITORY = SampleDataRepository()


def get_sample_repository() -> SampleDataRepository:
    return _SAMPLE_REPOSITORY
