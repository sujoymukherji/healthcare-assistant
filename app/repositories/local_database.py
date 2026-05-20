from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time
from pathlib import Path

from app.schemas.domain import ActorType, Appointment, AppointmentStatus, Doctor, DoctorAvailabilityWindow, MedicalRecordEntry, Patient, RecordEntryType, SourceReference
from app.utils.config import DATABASE_PATH


class LocalDatabaseRepository:
    """Simple SQLite repository for persisted patient profiles and medical records."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workbook_patients (
                    patient_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    date_of_birth TEXT,
                    phone TEXT,
                    gender TEXT,
                    address TEXT,
                    primary_conditions_json TEXT NOT NULL DEFAULT '[]',
                    allergies_json TEXT NOT NULL DEFAULT '[]',
                    chronic_conditions_json TEXT NOT NULL DEFAULT '[]',
                    preferred_doctors_json TEXT NOT NULL DEFAULT '[]',
                    source_path TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workbook_records (
                    record_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subjective TEXT,
                    objective TEXT,
                    assessment TEXT,
                    plan TEXT,
                    visit_date TEXT,
                    doctor_id TEXT,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    structured_fields_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sample_patients (
                    patient_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    date_of_birth TEXT,
                    phone TEXT,
                    gender TEXT,
                    address TEXT,
                    primary_conditions_json TEXT NOT NULL DEFAULT '[]',
                    allergies_json TEXT NOT NULL DEFAULT '[]',
                    chronic_conditions_json TEXT NOT NULL DEFAULT '[]',
                    preferred_doctors_json TEXT NOT NULL DEFAULT '[]',
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS doctors (
                    doctor_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    clinic_name TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    gender TEXT,
                    availability_json TEXT NOT NULL DEFAULT '[]',
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS medical_records (
                    record_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subjective TEXT,
                    objective TEXT,
                    assessment TEXT,
                    plan TEXT,
                    visit_date TEXT,
                    doctor_id TEXT,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    structured_fields_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    status TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    appointment_time TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    booked_by_actor TEXT NOT NULL,
                    booking_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    workflow_type TEXT,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    planner_output_json TEXT NOT NULL DEFAULT '{}',
                    final_workflow_type TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS system_errors (
                    error_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    retryable INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS langsmith_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    workflow_type TEXT,
                    trace_url TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        self._ensure_column('workbook_patients', 'date_of_birth', 'TEXT')
        self._ensure_column('workbook_patients', 'primary_conditions_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('workbook_patients', 'allergies_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('workbook_patients', 'chronic_conditions_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('workbook_patients', 'preferred_doctors_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('sample_patients', 'date_of_birth', 'TEXT')
        self._ensure_column('sample_patients', 'primary_conditions_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('sample_patients', 'allergies_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('sample_patients', 'chronic_conditions_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('sample_patients', 'preferred_doctors_json', "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column('doctors', 'availability_json', "TEXT NOT NULL DEFAULT '[]'")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        with self._connect() as connection:
            rows = connection.execute(f'PRAGMA table_info({table})').fetchall()
            columns = {row['name'] for row in rows}
            if column not in columns:
                try:
                    connection.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
                    connection.commit()
                except sqlite3.OperationalError as error:
                    if 'duplicate column name' not in str(error).lower():
                        raise

    def upsert_workbook_patient(self, patient: Patient, source_path: str) -> None:
        self._upsert_patient_into_table('workbook_patients', patient, source_path=source_path, source_type='workbook')

    def upsert_workbook_record(self, record: MedicalRecordEntry) -> None:
        self._upsert_record_into_table('workbook_records', record)

    def list_workbook_patients(self) -> list[Patient]:
        return self._list_patients_from_table('workbook_patients')

    def list_workbook_records(self) -> list[MedicalRecordEntry]:
        return self._list_records_from_table('workbook_records')

    def upsert_patient(self, patient: Patient) -> None:
        source_ref = patient.source_refs[0] if patient.source_refs else SourceReference(source_type='unknown', source_path='unknown')
        self._upsert_patient_into_table('sample_patients', patient, source_path=source_ref.source_path, source_type=source_ref.source_type)

    def upsert_record(self, record: MedicalRecordEntry) -> None:
        self._upsert_record_into_table('medical_records', record)

    def list_sample_patients(self) -> list[Patient]:
        return self._list_patients_from_table('sample_patients', include_source_type=True)

    def get_sample_patient(self, patient_id: str) -> Patient | None:
        patients = self._list_patients_from_table('sample_patients', include_source_type=True)
        return next((patient for patient in patients if patient.patient_id == patient_id), None)

    def list_medical_records(self) -> list[MedicalRecordEntry]:
        return self._list_records_from_table('medical_records')

    def count_sample_patients(self) -> int:
        return self._count_rows('sample_patients')

    def count_medical_records(self) -> int:
        return self._count_rows('medical_records')

    def count_doctors(self) -> int:
        return self._count_rows('doctors')

    def count_appointments(self) -> int:
        return self._count_rows('appointments')

    def clear_workbook_snapshot(self) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM workbook_records')
            connection.execute('DELETE FROM workbook_patients')
            connection.commit()

    def clear_sample_snapshot(self) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM medical_records')
            connection.execute('DELETE FROM doctors')
            connection.execute('DELETE FROM sample_patients')
            connection.commit()

    def clear_appointments(self) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM appointments')
            connection.commit()

    def clear_observability(self) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM interaction_logs')
            connection.execute('DELETE FROM planner_traces')
            connection.execute('DELETE FROM system_errors')
            connection.execute('DELETE FROM langsmith_runs')
            connection.commit()

    def delete_sample_patient(self, patient_id: str) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM sample_patients WHERE patient_id = ?', (patient_id,))
            connection.execute('DELETE FROM medical_records WHERE patient_id = ?', (patient_id,))
            connection.execute('DELETE FROM appointments WHERE patient_id = ?', (patient_id,))
            connection.commit()

    def upsert_doctor(self, doctor: Doctor) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO doctors (
                    doctor_id, full_name, specialty, clinic_name, location_id,
                    phone, email, gender, availability_json, active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doctor_id) DO UPDATE SET
                    full_name=excluded.full_name,
                    specialty=excluded.specialty,
                    clinic_name=excluded.clinic_name,
                    location_id=excluded.location_id,
                    phone=excluded.phone,
                    email=excluded.email,
                    gender=excluded.gender,
                    availability_json=excluded.availability_json,
                    active=excluded.active
                """,
                (
                    doctor.doctor_id,
                    doctor.full_name,
                    doctor.specialty,
                    doctor.clinic_name,
                    doctor.location_id,
                    doctor.phone,
                    doctor.email,
                    doctor.gender,
                    json.dumps([window.model_dump(mode='json') for window in doctor.availability]),
                    1 if doctor.active else 0,
                ),
            )
            connection.commit()

    def list_doctors(self) -> list[Doctor]:
        doctors: list[Doctor] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT doctor_id, full_name, specialty, clinic_name, location_id, phone, email, gender, availability_json, active
                FROM doctors
                ORDER BY full_name
                """
            ).fetchall()
        for row in rows:
            doctors.append(
                Doctor(
                    doctor_id=row['doctor_id'],
                    full_name=row['full_name'],
                    specialty=row['specialty'],
                    clinic_name=row['clinic_name'],
                    location_id=row['location_id'],
                    phone=row['phone'],
                    email=row['email'],
                    gender=row['gender'],
                    availability=[
                        DoctorAvailabilityWindow.model_validate(item)
                        for item in json.loads(row['availability_json'] or '[]')
                    ],
                    active=bool(row['active']),
                )
            )
        return doctors

    def upsert_appointment(self, appointment: Appointment) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO appointments (
                    appointment_id, patient_id, doctor_id, specialty, status, appointment_date,
                    appointment_time, location_id, booked_by_actor, booking_reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(appointment_id) DO UPDATE SET
                    patient_id=excluded.patient_id,
                    doctor_id=excluded.doctor_id,
                    specialty=excluded.specialty,
                    status=excluded.status,
                    appointment_date=excluded.appointment_date,
                    appointment_time=excluded.appointment_time,
                    location_id=excluded.location_id,
                    booked_by_actor=excluded.booked_by_actor,
                    booking_reason=excluded.booking_reason,
                    created_at=excluded.created_at
                """,
                (
                    appointment.appointment_id,
                    appointment.patient_id,
                    appointment.doctor_id,
                    appointment.specialty,
                    appointment.status.value,
                    appointment.appointment_date.isoformat(),
                    appointment.appointment_time.strftime('%H:%M:%S'),
                    appointment.location_id,
                    appointment.booked_by_actor.value,
                    appointment.booking_reason,
                    appointment.created_at.isoformat(),
                ),
            )
            connection.commit()

    def list_appointments(self) -> list[Appointment]:
        appointments: list[Appointment] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT appointment_id, patient_id, doctor_id, specialty, status, appointment_date,
                       appointment_time, location_id, booked_by_actor, booking_reason, created_at
                FROM appointments
                ORDER BY appointment_date, appointment_time, appointment_id
                """
            ).fetchall()
        for row in rows:
            appointments.append(
                Appointment(
                    appointment_id=row['appointment_id'],
                    patient_id=row['patient_id'],
                    doctor_id=row['doctor_id'],
                    specialty=row['specialty'],
                    status=AppointmentStatus(row['status']),
                    appointment_date=date.fromisoformat(row['appointment_date']),
                    appointment_time=time.fromisoformat(row['appointment_time']),
                    location_id=row['location_id'],
                    booked_by_actor=ActorType(row['booked_by_actor']),
                    booking_reason=row['booking_reason'],
                    created_at=datetime.fromisoformat(row['created_at']),
                )
            )
        return appointments

    def log_interaction(
        self,
        *,
        interaction_id: str,
        session_id: str,
        actor: str,
        workflow_type: str | None,
        user_message: str,
        assistant_message: str | None,
        context: dict[str, object] | None,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO interaction_logs (
                    interaction_id, session_id, actor, workflow_type, user_message,
                    assistant_message, context_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    session_id,
                    actor,
                    workflow_type,
                    user_message,
                    assistant_message,
                    json.dumps(context or {}),
                    created_at.isoformat(),
                ),
            )
            connection.commit()

    def log_planner_trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        actor: str,
        user_message: str,
        context: dict[str, object] | None,
        planner_output: dict[str, object] | None,
        final_workflow_type: str | None,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO planner_traces (
                    trace_id, session_id, actor, user_message, context_json,
                    planner_output_json, final_workflow_type, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    session_id,
                    actor,
                    user_message,
                    json.dumps(context or {}),
                    json.dumps(planner_output or {}),
                    final_workflow_type,
                    created_at.isoformat(),
                ),
            )
            connection.commit()

    def log_system_error(
        self,
        *,
        error_id: str,
        session_id: str,
        actor: str,
        stage: str,
        error_message: str,
        stack_trace: str | None,
        retryable: bool,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO system_errors (
                    error_id, session_id, actor, stage, error_message, stack_trace, retryable, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error_id,
                    session_id,
                    actor,
                    stage,
                    error_message,
                    stack_trace,
                    1 if retryable else 0,
                    created_at.isoformat(),
                ),
            )
            connection.commit()

    def log_langsmith_run(
        self,
        *,
        run_id: str,
        session_id: str,
        actor: str,
        workflow_type: str | None,
        trace_url: str | None,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO langsmith_runs (
                    run_id, session_id, actor, workflow_type, trace_url, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    actor,
                    workflow_type,
                    trace_url,
                    created_at.isoformat(),
                ),
            )
            connection.commit()

    def list_interaction_logs(self, *, actor: str | None = None, for_date: date | None = None) -> list[dict[str, object]]:
        query = """
            SELECT interaction_id, session_id, actor, workflow_type, user_message,
                   assistant_message, context_json, created_at
            FROM interaction_logs
        """
        filters: list[str] = []
        params: list[object] = []
        if actor:
            filters.append('actor = ?')
            params.append(actor)
        if for_date:
            filters.append('substr(created_at, 1, 10) = ?')
            params.append(for_date.isoformat())
        if filters:
            query += ' WHERE ' + ' AND '.join(filters)
        query += ' ORDER BY created_at DESC'
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                'Interaction ID': row['interaction_id'],
                'Session ID': row['session_id'],
                'Actor': row['actor'],
                'Workflow': row['workflow_type'] or '',
                'User Message': row['user_message'],
                'Assistant Message': row['assistant_message'] or '',
                'Context': json.loads(row['context_json'] or '{}'),
                'Created At': row['created_at'],
            }
            for row in rows
        ]

    def list_planner_traces(self, *, actor: str | None = None, for_date: date | None = None) -> list[dict[str, object]]:
        query = """
            SELECT trace_id, session_id, actor, user_message, context_json,
                   planner_output_json, final_workflow_type, created_at
            FROM planner_traces
        """
        filters: list[str] = []
        params: list[object] = []
        if actor:
            filters.append('actor = ?')
            params.append(actor)
        if for_date:
            filters.append('substr(created_at, 1, 10) = ?')
            params.append(for_date.isoformat())
        if filters:
            query += ' WHERE ' + ' AND '.join(filters)
        query += ' ORDER BY created_at DESC'
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                'Trace ID': row['trace_id'],
                'Session ID': row['session_id'],
                'Actor': row['actor'],
                'User Message': row['user_message'],
                'Planner Output': json.loads(row['planner_output_json'] or '{}'),
                'Final Workflow': row['final_workflow_type'] or '',
                'Context': json.loads(row['context_json'] or '{}'),
                'Created At': row['created_at'],
            }
            for row in rows
        ]

    def list_system_errors(self, *, actor: str | None = None, for_date: date | None = None) -> list[dict[str, object]]:
        query = """
            SELECT error_id, session_id, actor, stage, error_message, stack_trace, retryable, created_at
            FROM system_errors
        """
        filters: list[str] = []
        params: list[object] = []
        if actor:
            filters.append('actor = ?')
            params.append(actor)
        if for_date:
            filters.append('substr(created_at, 1, 10) = ?')
            params.append(for_date.isoformat())
        if filters:
            query += ' WHERE ' + ' AND '.join(filters)
        query += ' ORDER BY created_at DESC'
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                'Error ID': row['error_id'],
                'Session ID': row['session_id'],
                'Actor': row['actor'],
                'Stage': row['stage'],
                'Error Message': row['error_message'],
                'Retryable': bool(row['retryable']),
                'Created At': row['created_at'],
                'Stack Trace': row['stack_trace'] or '',
            }
            for row in rows
        ]

    def list_langsmith_runs(self, *, actor: str | None = None, for_date: date | None = None) -> list[dict[str, object]]:
        query = """
            SELECT run_id, session_id, actor, workflow_type, trace_url, created_at
            FROM langsmith_runs
        """
        filters: list[str] = []
        params: list[object] = []
        if actor:
            filters.append('actor = ?')
            params.append(actor)
        if for_date:
            filters.append('substr(created_at, 1, 10) = ?')
            params.append(for_date.isoformat())
        if filters:
            query += ' WHERE ' + ' AND '.join(filters)
        query += ' ORDER BY created_at DESC'
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                'Run ID': row['run_id'],
                'Session ID': row['session_id'],
                'Actor': row['actor'],
                'Workflow': row['workflow_type'] or '',
                'Trace URL': row['trace_url'] or '',
                'Created At': row['created_at'],
            }
            for row in rows
        ]

    def _count_rows(self, table: str) -> int:
        with self._connect() as connection:
            row = connection.execute(f'SELECT COUNT(*) AS count FROM {table}').fetchone()
        return int(row['count']) if row else 0

    def _upsert_patient_into_table(self, table: str, patient: Patient, *, source_path: str, source_type: str) -> None:
        values = (
            patient.patient_id,
            patient.full_name,
            patient.date_of_birth.isoformat() if patient.date_of_birth else None,
            patient.phone,
            patient.gender,
            patient.address,
            json.dumps(patient.primary_conditions or []),
            json.dumps(patient.allergies or []),
            json.dumps(patient.chronic_conditions or []),
            json.dumps(patient.preferred_doctors or []),
            source_path,
        )
        with self._connect() as connection:
            if table == 'sample_patients':
                connection.execute(
                    """
                    INSERT INTO sample_patients (
                        patient_id, full_name, date_of_birth, phone, gender, address,
                        primary_conditions_json, allergies_json, chronic_conditions_json,
                        preferred_doctors_json, source_path, source_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        full_name=excluded.full_name,
                        date_of_birth=excluded.date_of_birth,
                        phone=excluded.phone,
                        gender=excluded.gender,
                        address=excluded.address,
                        primary_conditions_json=excluded.primary_conditions_json,
                        allergies_json=excluded.allergies_json,
                        chronic_conditions_json=excluded.chronic_conditions_json,
                        preferred_doctors_json=excluded.preferred_doctors_json,
                        source_path=excluded.source_path,
                        source_type=excluded.source_type
                    """,
                    (*values, source_type),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO workbook_patients (
                        patient_id, full_name, date_of_birth, phone, gender, address,
                        primary_conditions_json, allergies_json, chronic_conditions_json,
                        preferred_doctors_json, source_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        full_name=excluded.full_name,
                        date_of_birth=excluded.date_of_birth,
                        phone=excluded.phone,
                        gender=excluded.gender,
                        address=excluded.address,
                        primary_conditions_json=excluded.primary_conditions_json,
                        allergies_json=excluded.allergies_json,
                        chronic_conditions_json=excluded.chronic_conditions_json,
                        preferred_doctors_json=excluded.preferred_doctors_json,
                        source_path=excluded.source_path
                    """,
                    values,
                )
            connection.commit()

    def _upsert_record_into_table(self, table: str, record: MedicalRecordEntry) -> None:
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO {table} (
                    record_id, patient_id, entry_type, title, subjective, objective, assessment, plan,
                    visit_date, doctor_id, source_type, source_path, structured_fields_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    patient_id=excluded.patient_id,
                    entry_type=excluded.entry_type,
                    title=excluded.title,
                    subjective=excluded.subjective,
                    objective=excluded.objective,
                    assessment=excluded.assessment,
                    plan=excluded.plan,
                    visit_date=excluded.visit_date,
                    doctor_id=excluded.doctor_id,
                    source_type=excluded.source_type,
                    source_path=excluded.source_path,
                    structured_fields_json=excluded.structured_fields_json
                """,
                (
                    record.record_id,
                    record.patient_id,
                    record.entry_type.value,
                    record.title,
                    record.subjective,
                    record.objective,
                    record.assessment,
                    record.plan,
                    record.visit_date.isoformat() if record.visit_date else None,
                    record.doctor_id,
                    record.source_type,
                    record.source_path,
                    json.dumps(record.structured_fields or {}),
                ),
            )
            connection.commit()

    def _list_patients_from_table(self, table: str, *, include_source_type: bool = False) -> list[Patient]:
        patients: list[Patient] = []
        columns = (
            'patient_id, full_name, date_of_birth, phone, gender, address, '
            'primary_conditions_json, allergies_json, chronic_conditions_json, preferred_doctors_json, source_path'
        ) + (', source_type' if include_source_type else '')
        with self._connect() as connection:
            rows = connection.execute(f'SELECT {columns} FROM {table} ORDER BY patient_id').fetchall()
        for row in rows:
            source_type = row['source_type'] if include_source_type else 'workbook'
            patients.append(
                Patient(
                    patient_id=row['patient_id'],
                    full_name=row['full_name'],
                    date_of_birth=date.fromisoformat(row['date_of_birth']) if row['date_of_birth'] else None,
                    phone=row['phone'],
                    gender=row['gender'],
                    address=row['address'],
                    primary_conditions=json.loads(row['primary_conditions_json'] or '[]'),
                    allergies=json.loads(row['allergies_json'] or '[]'),
                    chronic_conditions=json.loads(row['chronic_conditions_json'] or '[]'),
                    preferred_doctors=json.loads(row['preferred_doctors_json'] or '[]'),
                    source_refs=[SourceReference(source_type=source_type, source_path=row['source_path'])],
                )
            )
        return patients

    def _list_records_from_table(self, table: str) -> list[MedicalRecordEntry]:
        records: list[MedicalRecordEntry] = []
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT record_id, patient_id, entry_type, title, subjective, objective, assessment, plan,
                       visit_date, doctor_id, source_type, source_path, structured_fields_json
                FROM {table}
                ORDER BY record_id
                """
            ).fetchall()
        for row in rows:
            visit_date = date.fromisoformat(row['visit_date']) if row['visit_date'] else None
            records.append(
                MedicalRecordEntry(
                    record_id=row['record_id'],
                    patient_id=row['patient_id'],
                    entry_type=RecordEntryType(row['entry_type']),
                    visit_date=visit_date,
                    title=row['title'],
                    subjective=row['subjective'],
                    objective=row['objective'],
                    assessment=row['assessment'],
                    plan=row['plan'],
                    doctor_id=row['doctor_id'],
                    source_type=row['source_type'],
                    source_path=row['source_path'],
                    structured_fields=json.loads(row['structured_fields_json'] or '{}'),
                )
            )
        return records


_LOCAL_DATABASE = LocalDatabaseRepository()


def get_local_database_repository() -> LocalDatabaseRepository:
    return _LOCAL_DATABASE
