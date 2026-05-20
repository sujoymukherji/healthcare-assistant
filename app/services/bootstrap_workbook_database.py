from __future__ import annotations

import argparse
from pathlib import Path

from app.repositories.local_database import get_local_database_repository
from app.repositories.sample_data_repository import SampleDataRepository
from app.services.sample_data_loader import SampleDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description='Bootstrap workbook-backed patients and summary records.')
    parser.add_argument('--workbook', type=Path, help='Path to a workbook following the records.xlsx template.')
    args = parser.parse_args()

    loader = SampleDataLoader(workbook_path=args.workbook)
    repository = SampleDataRepository(loader=loader)
    database = get_local_database_repository()

    database.clear_workbook_snapshot()
    for patient in repository.workbook_patients:
        source_path = patient.source_refs[0].source_path if patient.source_refs else 'unknown'
        database.upsert_workbook_patient(patient, source_path)
    for record in repository.workbook_records:
        database.upsert_workbook_record(record)

    print({
        'db_path': str(database.db_path),
        'workbook_patients': len(database.list_workbook_patients()),
        'workbook_records': len(database.list_workbook_records()),
    })


if __name__ == '__main__':
    main()
