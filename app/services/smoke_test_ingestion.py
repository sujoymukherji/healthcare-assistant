from __future__ import annotations

import json

from app.services.sample_data_loader import SampleDataLoader


def main() -> None:
    loader = SampleDataLoader()
    bundle = loader.load_all()
    payload = {
        "patient_count": len(bundle.patients),
        "record_count": len(bundle.records),
        "diagnosis_count": len(bundle.diagnoses),
        "raw_spreadsheet_row_count": len(bundle.raw_spreadsheet_rows),
        "raw_pdf_count": len(bundle.raw_pdf_documents),
        "patients": bundle.patients,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
