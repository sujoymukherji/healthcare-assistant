from __future__ import annotations

from pydantic import BaseModel, Field


class RawSpreadsheetRow(BaseModel):
    source_file: str
    sheet_name: str
    row_index: int
    raw_cells: dict[str, str] = Field(default_factory=dict)


class RawPdfDocument(BaseModel):
    source_file: str
    source_path: str
    document_type: str
    page_count: int
    extracted_text: str


class IngestedPatientBundle(BaseModel):
    patients: list[dict[str, object]] = Field(default_factory=list)
    doctors: list[dict[str, object]] = Field(default_factory=list)
    records: list[dict[str, object]] = Field(default_factory=list)
    diagnoses: list[dict[str, object]] = Field(default_factory=list)
    workbook_patients: list[dict[str, object]] = Field(default_factory=list)
    workbook_records: list[dict[str, object]] = Field(default_factory=list)
    raw_spreadsheet_rows: list[RawSpreadsheetRow] = Field(default_factory=list)
    raw_pdf_documents: list[RawPdfDocument] = Field(default_factory=list)
