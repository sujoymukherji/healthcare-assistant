from __future__ import annotations

import argparse
from pathlib import Path

from app.repositories.appointment_repository import get_appointment_repository
from app.repositories.local_database import get_local_database_repository
from app.repositories.sample_data_repository import SampleDataRepository
from app.schemas.domain import Doctor, MedicalRecordEntry, Patient
from app.services.sample_data_loader import SampleDataLoader
from app.utils.config import CHROMA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Bootstrap SQLite, appointment seed data, and Chroma collections for the refactored healthcare assistant.'
    )
    parser.add_argument('--workbook', type=Path, help='Path to a workbook following the records.xlsx template.')
    parser.add_argument('--pdf-dir', type=Path, help='Directory containing sample_*.pdf files for patient history.')
    parser.add_argument('--skip-chroma', action='store_true', help='Skip rebuilding the Chroma vector collections.')
    args = parser.parse_args()

    loader = SampleDataLoader(samples_dir=args.pdf_dir, workbook_path=args.workbook)
    database = get_local_database_repository()
    appointments = get_appointment_repository()
    bundle = loader.load_all()

    database.clear_observability()
    database.clear_sample_snapshot()
    database.clear_workbook_snapshot()
    for doctor in bundle.doctors:
        database.upsert_doctor(Doctor.model_validate(doctor))
    for patient in bundle.patients:
        database.upsert_patient(Patient.model_validate(patient))
    for record in bundle.records:
        database.upsert_record(MedicalRecordEntry.model_validate(record))
    for patient in bundle.workbook_patients:
        patient_model = Patient.model_validate(patient)
        source_path = patient_model.source_refs[0].source_path if patient_model.source_refs else 'unknown'
        database.upsert_workbook_patient(patient_model, source_path)
    for record in bundle.workbook_records:
        database.upsert_workbook_record(MedicalRecordEntry.model_validate(record))

    seeded_appointments = appointments.reset_demo_schedule()
    chroma_counts: dict[str, int] | None = None
    if not args.skip_chroma:
        from app.services.chroma_ingestion import ChromaIngestionService

        repository = SampleDataRepository(loader=loader)
        chroma_counts = ChromaIngestionService(repository=repository).ingest_all(reset=True)

    print(
        {
            'db_path': str(database.db_path),
            'sample_patients': database.count_sample_patients(),
            'doctors': database.count_doctors(),
            'medical_records': database.count_medical_records(),
            'workbook_patients': len(database.list_workbook_patients()),
            'workbook_records': len(database.list_workbook_records()),
            'appointments': len(seeded_appointments),
            'chroma_dir': str(CHROMA_DIR),
            'chroma_counts': chroma_counts,
        }
    )


if __name__ == '__main__':
    main()
