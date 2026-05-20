from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfReader

from app.schemas.domain import Diagnosis, Doctor, DoctorAvailabilityWindow, MedicalRecordEntry, Patient, RecordEntryType, SourceReference
from app.schemas.raw import IngestedPatientBundle, RawPdfDocument, RawSpreadsheetRow
from app.services.patient_summary_parser import PatientSummaryInsights, PatientSummaryParser
from app.services.record_summary_generator import record_summary_generator
from app.utils.config import SAMPLES_DIR

_XLSX_NS = {
    'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
_SECTION_PATTERNS = {
    'subjective': [r'Subjective Notes:?'],
    'objective': [r'Objective Notes:?'],
    'assessment': [r'Assessment Notes:?', r'DIAGNOSIS:?'],
    'plan': [r'Plan Notes:?'],
}
_SIMPLE_FIELD_PATTERNS = {
    'patient_name': [r'Patient:\s*(.+)'],
    'dob': [r'DOB:\s*(.+)'],
    'gender': [r'Gender:\s*(.+)'],
    'phone': [r'Phone:\s*(.+)'],
    'address': [r'Address:\s*(.+)'],
    'visit_date': [r'Visit Date:\s*(.+)', r'^\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)\s*$'],
    'location': [r'Location:\s*(.+)'],
}
_VISIT_SPLIT_RE = re.compile(r'(?im)^Visit Date:\s*.+$')


class SampleDataLoader:
    """Loads local workbook and PDF data into canonical demo objects."""

    def __init__(
        self,
        samples_dir: Path | None = None,
        workbook_path: Path | None = None,
        doctor_workbook_path: Path | None = None,
        pdf_glob: str = 'sample_*.pdf',
        summary_parser: PatientSummaryParser | None = None,
    ) -> None:
        self.samples_dir = samples_dir or SAMPLES_DIR
        self.workbook_path = workbook_path
        self.doctor_workbook_path = doctor_workbook_path
        self.pdf_glob = pdf_glob
        self.summary_parser = summary_parser or PatientSummaryParser()

    def load_all(
        self,
        workbook_path: Path | None = None,
        pdf_paths: list[Path] | None = None,
    ) -> IngestedPatientBundle:
        workbook = workbook_path or self.workbook_path or (self.samples_dir / 'records.xlsx')
        doctor_workbook = self.doctor_workbook_path or (SAMPLES_DIR / 'Doctors.xlsx')
        raw_rows = self._load_workbook_rows(workbook)
        raw_doctor_rows = self._load_workbook_rows(doctor_workbook)
        raw_pdfs = self._load_pdf_documents(pdf_paths=pdf_paths)
        doctors = self._normalize_doctor_rows(raw_doctor_rows)
        workbook_patients, workbook_records = self._normalize_workbook_rows(raw_rows)
        patients_by_id = {patient.patient_id: deepcopy(patient) for patient in workbook_patients}
        patients_by_name = {self._normalize_name(patient.full_name): patient for patient in patients_by_id.values() if patient.full_name}
        patients_by_phone = {self._normalize_phone(patient.phone): patient for patient in patients_by_id.values() if self._normalize_phone(patient.phone)}
        pdf_records, diagnoses = self._normalize_pdfs(raw_pdfs, patients_by_id, patients_by_name, patients_by_phone)
        patients = sorted(patients_by_id.values(), key=lambda patient: patient.patient_id)
        records = sorted([*workbook_records, *pdf_records], key=lambda record: ((record.visit_date.isoformat() if record.visit_date else ''), record.record_id))
        return IngestedPatientBundle(
            patients=[patient.model_dump(mode='json') for patient in patients],
            doctors=[doctor.model_dump(mode='json') for doctor in doctors],
            records=[record.model_dump(mode='json') for record in records],
            diagnoses=[diagnosis.model_dump(mode='json') for diagnosis in diagnoses],
            workbook_patients=[patient.model_dump(mode='json') for patient in workbook_patients],
            workbook_records=[record.model_dump(mode='json') for record in workbook_records],
            raw_spreadsheet_rows=raw_rows,
            raw_pdf_documents=raw_pdfs,
        )

    def _load_workbook_rows(self, workbook_path: Path) -> list[RawSpreadsheetRow]:
        if not workbook_path.exists():
            return []
        rows: list[RawSpreadsheetRow] = []
        with ZipFile(workbook_path) as archive:
            shared_strings = self._read_shared_strings(archive)
            workbook_root = ET.fromstring(archive.read('xl/workbook.xml'))
            rels_root = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
            rel_map = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels_root}
            for sheet in workbook_root.findall('a:sheets/a:sheet', _XLSX_NS):
                sheet_name = sheet.attrib['name']
                rel_id = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
                target = rel_map[rel_id].lstrip('/')
                if not target.startswith('xl/'):
                    target = posixpath.normpath(posixpath.join('xl', target))
                sheet_root = ET.fromstring(archive.read(target))
                for row in sheet_root.findall('.//a:sheetData/a:row', _XLSX_NS):
                    row_cells: dict[str, str] = {}
                    for cell in row.findall('a:c', _XLSX_NS):
                        ref = cell.attrib.get('r', '')
                        col = re.sub(r'\d', '', ref) or 'UNKNOWN'
                        cell_type = cell.attrib.get('t')
                        value_node = cell.find('a:v', _XLSX_NS)
                        inline_node = cell.find('a:is', _XLSX_NS)
                        value = '' if value_node is None else value_node.text or ''
                        if cell_type == 's' and value:
                            value = shared_strings[int(value)]
                        elif cell_type == 'inlineStr' and inline_node is not None:
                            value = ''.join(node.text or '' for node in inline_node.findall('.//a:t', _XLSX_NS))
                        if value:
                            row_cells[col] = value
                    if row_cells:
                        rows.append(
                            RawSpreadsheetRow(
                                source_file=workbook_path.name,
                                sheet_name=sheet_name,
                                row_index=int(row.attrib.get('r', '0')),
                                raw_cells=row_cells,
                            )
                        )
        return rows

    def _normalize_workbook_rows(self, rows: list[RawSpreadsheetRow]) -> tuple[list[Patient], list[MedicalRecordEntry]]:
        if not rows:
            return [], []
        header_row = min(rows, key=lambda row: row.row_index)
        headers = {col: self._normalize_header_name(value) for col, value in header_row.raw_cells.items()}
        source_path = str(self.samples_dir / header_row.source_file)
        patients: dict[str, Patient] = {}
        records: list[MedicalRecordEntry] = []
        for row in rows:
            if row.row_index == header_row.row_index:
                continue
            mapped = {headers.get(col, col.lower()): value.strip() for col, value in row.raw_cells.items()}
            full_name = mapped.get('name', '').strip()
            phone = self._normalize_phone(mapped.get('phone_number') or mapped.get('phone'))
            if not full_name and not phone:
                continue
            patient_id = self._make_patient_id(full_name or phone or f'row_{row.row_index}')
            patient = patients.get(patient_id)
            if patient is None:
                patient = Patient(
                    patient_id=patient_id,
                    full_name=full_name or 'Workbook Patient',
                    gender=mapped.get('gender') or None,
                    phone=phone,
                    address=mapped.get('address') or None,
                    source_refs=[SourceReference(source_type='workbook', source_path=source_path)],
                )
                patients[patient_id] = patient
            else:
                if full_name and patient.full_name == 'Workbook Patient':
                    patient.full_name = full_name
                if phone and not patient.phone:
                    patient.phone = phone
                if mapped.get('gender') and not patient.gender:
                    patient.gender = mapped.get('gender')
                if mapped.get('address') and not patient.address:
                    patient.address = mapped.get('address')
                self._append_source_ref(patient, 'workbook', source_path)

            summary = mapped.get('summary', '').strip()
            if summary:
                insights = self.summary_parser.parse_summary(summary)
                self._merge_patient_summary(patient, insights)
                records.append(
                    MedicalRecordEntry(
                        record_id=f'rec_workbook_{row.row_index:03d}',
                        patient_id=patient.patient_id,
                        entry_type=RecordEntryType.HISTORY_SUMMARY,
                        title='Workbook Summary',
                        subjective=summary,
                        source_type='workbook',
                        source_path=source_path,
                        structured_fields={
                            'origin': 'records.xlsx',
                            'primary_conditions': insights.primary_conditions,
                            'allergies': insights.allergies,
                            'chronic_conditions': insights.chronic_conditions,
                            'visit_summary': self._summarize_workbook_entry(insights, summary),
                        },
                    )
                )
        return sorted(patients.values(), key=lambda patient: patient.patient_id), records

    def _normalize_doctor_rows(self, rows: list[RawSpreadsheetRow]) -> list[Doctor]:
        if not rows:
            return []
        header_row = min(rows, key=lambda row: row.row_index)
        headers = {col: self._normalize_header_name(value) for col, value in header_row.raw_cells.items()}
        doctors: dict[str, Doctor] = {}
        for row in rows:
            if row.row_index == header_row.row_index:
                continue
            mapped = {headers.get(col, col.lower()): value.strip() for col, value in row.raw_cells.items()}
            full_name = mapped.get('name', '').strip()
            if not full_name:
                continue
            specialty = self._normalize_specialty(mapped.get('specialization', '').strip() or 'General Medicine')
            doctor = Doctor(
                doctor_id=self._make_doctor_id(full_name),
                full_name=full_name,
                specialty=specialty,
                clinic_name='Apollo Health',
                location_id='clinic_main',
                phone=self._normalize_phone(mapped.get('phone_number') or mapped.get('phone')),
                email=mapped.get('email') or None,
                gender=mapped.get('gender') or None,
                active=True,
                availability=self._parse_doctor_availability(mapped.get('availability') or ''),
            )
            doctors[doctor.doctor_id] = doctor
        return sorted(doctors.values(), key=lambda doctor: doctor.full_name.lower())

    def _parse_doctor_availability(self, raw_value: str) -> list[DoctorAvailabilityWindow]:
        if not raw_value.strip():
            return []
        windows: list[DoctorAvailabilityWindow] = []
        for segment in [part.strip() for part in raw_value.split(';') if part.strip()]:
            if '|' not in segment:
                continue
            day_text, time_text = [part.strip() for part in segment.split('|', 1)]
            days = self._parse_availability_days(day_text)
            time_window = self._parse_availability_time_range(time_text)
            if not days or time_window is None:
                continue
            start_time, end_time = time_window
            windows.append(
                DoctorAvailabilityWindow(
                    days_of_week=days,
                    start_time=start_time,
                    end_time=end_time,
                    source_text=segment,
                )
            )
        return windows

    def _parse_availability_days(self, raw_days: str) -> list[int]:
        day_lookup = {
            'mon': 0,
            'tue': 1,
            'wed': 2,
            'thu': 3,
            'fri': 4,
            'sat': 5,
            'sun': 6,
        }
        normalized = raw_days.strip().lower()
        if '-' in normalized:
            start_label, end_label = [part.strip()[:3] for part in normalized.split('-', 1)]
            if start_label in day_lookup and end_label in day_lookup:
                start_idx = day_lookup[start_label]
                end_idx = day_lookup[end_label]
                if start_idx <= end_idx:
                    return list(range(start_idx, end_idx + 1))
        values: list[int] = []
        for part in [piece.strip()[:3] for piece in normalized.split(',') if piece.strip()]:
            if part in day_lookup and day_lookup[part] not in values:
                values.append(day_lookup[part])
        return values

    def _parse_availability_time_range(self, raw_range: str):
        match = re.search(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', raw_range)
        if not match:
            return None
        start = datetime.strptime(match.group(1), '%H:%M').time()
        end = datetime.strptime(match.group(2), '%H:%M').time()
        return start, end


    def _summarize_workbook_entry(self, insights: PatientSummaryInsights, summary: str) -> str:
        conditions = insights.primary_conditions or insights.chronic_conditions
        if conditions:
            condition_text = ', '.join(conditions[:2])
            return f'Summary on file for {condition_text}.'
        cleaned = ' '.join(summary.split())
        if not cleaned:
            return ''
        snippet = cleaned[:117].rstrip() + '...' if len(cleaned) > 120 else cleaned
        return snippet

    def _read_shared_strings(self, archive: ZipFile) -> list[str]:
        if 'xl/sharedStrings.xml' not in archive.namelist():
            return []
        root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
        strings: list[str] = []
        for item in root.findall('a:si', _XLSX_NS):
            strings.append(''.join(node.text or '' for node in item.findall('.//a:t', _XLSX_NS)))
        return strings

    def _load_pdf_documents(self, pdf_paths: list[Path] | None = None) -> list[RawPdfDocument]:
        documents: list[RawPdfDocument] = []
        paths = pdf_paths or sorted(self.samples_dir.glob(self.pdf_glob))
        for path in paths:
            reader = PdfReader(str(path))
            text = '\n'.join((page.extract_text() or '') for page in reader.pages)
            documents.append(
                RawPdfDocument(
                    source_file=path.name,
                    source_path=str(path),
                    document_type=self._detect_document_type(path.name, text),
                    page_count=len(reader.pages),
                    extracted_text=text,
                )
            )
        return documents

    def _detect_document_type(self, filename: str, text: str) -> str:
        lowered = f'{filename} {text[:600]}'.lower()
        if 'history and physical note' in lowered:
            return 'history_and_physical_note'
        if 'subjective notes' in lowered and 'objective notes' in lowered:
            return 'patient_visit_summary'
        return 'generic_medical_document'

    def _normalize_pdfs(
        self,
        documents: list[RawPdfDocument],
        patients_by_id: dict[str, Patient],
        patients_by_name: dict[str, Patient],
        patients_by_phone: dict[str, Patient],
    ) -> tuple[list[MedicalRecordEntry], list[Diagnosis]]:
        records: list[MedicalRecordEntry] = []
        diagnoses: list[Diagnosis] = []
        diagnosis_index = 1
        for doc_index, document in enumerate(documents, start=1):
            visit_chunks = self._split_pdf_into_visit_chunks(document.extracted_text)
            for visit_index, chunk in enumerate(visit_chunks, start=1):
                parsed = self._parse_pdf_text(chunk)
                patient = self._resolve_patient_for_pdf(parsed, document, patients_by_id, patients_by_name, patients_by_phone)
                if patient is None:
                    continue
                self._update_patient_from_pdf(patient, parsed, chunk, document.source_path)
                assessment = parsed.get('assessment')
                condition_name, condition_code = self._split_assessment(assessment)
                if condition_name:
                    self._append_unique(patient.primary_conditions, condition_name)
                    if self._looks_chronic(condition_name):
                        self._append_unique(patient.chronic_conditions, condition_name)
                for allergy in self._extract_allergies(chunk):
                    self._append_unique(patient.allergies, allergy)
                plan_text = parsed.get('plan') or None
                structured_fields = self._extract_structured_fields(assessment, chunk)
                visit_summary = record_summary_generator.summarize_visit(assessment, plan_text)
                if visit_summary:
                    structured_fields['visit_summary'] = visit_summary
                record = MedicalRecordEntry(
                    record_id=f'rec_{doc_index:03d}_{visit_index:02d}',
                    patient_id=patient.patient_id,
                    entry_type=RecordEntryType.VISIT_NOTE,
                    visit_date=self._parse_date(parsed.get('visit_date')),
                    title=parsed.get('title') or document.document_type.replace('_', ' ').title(),
                    subjective=parsed.get('subjective') or None,
                    objective=parsed.get('objective') or None,
                    assessment=assessment or None,
                    plan=plan_text,
                    source_type='pdf',
                    source_path=document.source_path,
                    structured_fields=structured_fields,
                )
                records.append(record)
                diagnosis = self._build_diagnosis(
                    record.record_id,
                    patient.patient_id,
                    condition_name,
                    condition_code,
                    parsed.get('visit_date'),
                    diagnosis_index,
                )
                if diagnosis is not None:
                    diagnoses.append(diagnosis)
                    diagnosis_index += 1
        return records, diagnoses

    def _resolve_patient_for_pdf(
        self,
        parsed: dict[str, str],
        document: RawPdfDocument,
        patients_by_id: dict[str, Patient],
        patients_by_name: dict[str, Patient],
        patients_by_phone: dict[str, Patient],
    ) -> Patient | None:
        patient_name = parsed.get('patient_name') or self._infer_patient_name_from_text(document.extracted_text)
        normalized_name = self._normalize_name(patient_name)
        phone = self._normalize_phone(parsed.get('phone'))
        patient = None
        if phone:
            patient = patients_by_phone.get(phone)
        if patient is None and normalized_name:
            patient = patients_by_name.get(normalized_name)
        if patient is not None:
            return patient
        if not patient_name and not phone:
            return None
        patient = Patient(
            patient_id=self._make_patient_id(patient_name or phone or document.source_file),
            full_name=patient_name or 'PDF Patient',
            date_of_birth=self._parse_date(parsed.get('dob')),
            gender=parsed.get('gender') or None,
            phone=phone,
            address=parsed.get('address') or None,
            source_refs=[SourceReference(source_type='pdf', source_path=document.source_path)],
        )
        patients_by_id[patient.patient_id] = patient
        if normalized_name:
            patients_by_name[normalized_name] = patient
        if phone:
            patients_by_phone[phone] = patient
        return patient

    def _update_patient_from_pdf(self, patient: Patient, parsed: dict[str, str], chunk: str, source_path: str) -> None:
        patient.full_name = patient.full_name or parsed.get('patient_name') or 'PDF Patient'
        if parsed.get('dob') and not patient.date_of_birth:
            patient.date_of_birth = self._parse_date(parsed.get('dob'))
        if parsed.get('gender') and not patient.gender:
            patient.gender = parsed.get('gender')
        parsed_phone = self._normalize_phone(parsed.get('phone'))
        if parsed_phone and not patient.phone:
            patient.phone = parsed_phone
        if parsed.get('address') and not patient.address:
            patient.address = parsed.get('address')
        self._append_source_ref(patient, 'pdf', source_path)

        for condition in self._extract_background_conditions(chunk):
            self._append_unique(patient.primary_conditions, condition)
            if self._looks_chronic(condition):
                self._append_unique(patient.chronic_conditions, condition)
        for allergy in self._extract_allergies(chunk):
            self._append_unique(patient.allergies, allergy)

    def _split_pdf_into_visit_chunks(self, text: str) -> list[str]:
        matches = list(_VISIT_SPLIT_RE.finditer(text))
        if len(matches) <= 1:
            return [text]
        chunks: list[str] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            prefix_start = text.rfind('\n', 0, start)
            prefix_start = 0 if prefix_start == -1 else prefix_start + 1
            chunk = text[prefix_start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks or [text]

    def _parse_pdf_text(self, text: str) -> dict[str, str]:
        cleaned = re.sub(r'\r', '', text)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        result: dict[str, str] = {'title': lines[0] if lines else 'Medical Record'}
        for key, patterns in _SIMPLE_FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, cleaned, flags=re.MULTILINE)
                if match:
                    result[key] = match.group(1).strip()
                    break
        if not result.get('patient_name'):
            inferred = self._infer_patient_name_from_text(cleaned)
            if inferred:
                result['patient_name'] = inferred
        section_order = list(_SECTION_PATTERNS.keys())
        for index, key in enumerate(section_order):
            patterns = _SECTION_PATTERNS[key]
            next_patterns = []
            for later_key in section_order[index + 1:]:
                next_patterns.extend(_SECTION_PATTERNS[later_key])
            result[key] = self._extract_section(cleaned, patterns, next_patterns)
        return result

    def _infer_patient_name_from_text(self, text: str) -> str | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if re.fullmatch(r'[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+', first_line):
            return first_line
        return None

    def _extract_section(self, text: str, patterns: list[str], next_patterns: list[str]) -> str:
        start_match = None
        for pattern in patterns:
            start_match = re.search(pattern, text, flags=re.IGNORECASE)
            if start_match:
                break
        if not start_match:
            return ''
        start = start_match.end()
        end = len(text)
        for label in next_patterns:
            match = re.search(label, text[start:], flags=re.IGNORECASE)
            if match:
                end = min(end, start + match.start())
        section = text[start:end]
        return ' '.join(section.split())

    def _extract_structured_fields(self, assessment: str | None, text: str) -> dict[str, object]:
        fields: dict[str, object] = {}
        _, code = self._split_assessment(assessment)
        if code:
            fields['icd10_codes'] = [code]
        background_conditions = self._extract_background_conditions(text)
        if background_conditions:
            fields['background_conditions'] = background_conditions
        return fields

    def _split_assessment(self, assessment: str | None) -> tuple[str | None, str | None]:
        if not assessment:
            return None, None
        cleaned = assessment.replace('Diagnosis:', '').replace('DIAGNOSIS:', '').strip()
        code_match = re.search(r'(.+?)\s*\(([A-Z]\d+(?:\.\d+)?)\)', cleaned)
        if code_match:
            return code_match.group(1).strip(), code_match.group(2)
        short = cleaned.split('[')[0].strip()
        return short or None, None

    def _build_diagnosis(
        self,
        record_id: str,
        patient_id: str,
        condition_name: str | None,
        condition_code: str | None,
        visit_date: str | None,
        index: int,
    ) -> Diagnosis | None:
        if not condition_name:
            return None
        return Diagnosis(
            diagnosis_id=f'diag_{index:03d}',
            patient_id=patient_id,
            record_id=record_id,
            name=condition_name,
            code_system='ICD-10' if condition_code else None,
            code=condition_code,
            status='active',
            diagnosed_on=self._parse_date(visit_date),
        )

    def _extract_background_conditions(self, text: str) -> list[str]:
        conditions: list[str] = []
        for pattern in (r'PMH:\s*([^\n]+)', r'\bhistory of\s+([A-Za-z0-9 ,/-]+?)(?:[.;\n]|$)'):
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                fragment = match.group(1)
                for item in re.split(r',|;|/| and ', fragment):
                    cleaned = item.strip(' .')
                    if not cleaned:
                        continue
                    cleaned = re.sub(r'^a\s+', '', cleaned, flags=re.IGNORECASE)
                    cleaned = re.split(r'\b(?:fh|family history|meds|medications|concerns today|last pap)\b\s*:?\s*', cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(' .')
                    if len(cleaned) < 3:
                        continue
                    lowered_cleaned = cleaned.lower()
                    if any(token in lowered_cleaned for token in ('present illness', 'performed the physical exam', 'medical decision making')):
                        continue
                    label = cleaned.upper() if cleaned.upper() == 'PCOS' else cleaned.title()
                    self._append_unique(conditions, label)
        return conditions

    def _extract_allergies(self, text: str) -> list[str]:
        match = re.search(r'Allerg(?:y|ies):\s*([^.\n]+)', text, flags=re.IGNORECASE)
        if not match:
            return []
        allergies = [item.strip(' .') for item in re.split(r',|;|/| and ', match.group(1)) if item.strip(' .')]
        if any(item.lower() in {'none', 'nkda', 'no known allergies'} for item in allergies):
            return []
        return allergies

    def _merge_patient_summary(self, patient: Patient, insights: PatientSummaryInsights) -> None:
        for item in insights.primary_conditions:
            self._append_unique(patient.primary_conditions, item)
        for item in insights.allergies:
            self._append_unique(patient.allergies, item)
        for item in insights.chronic_conditions:
            self._append_unique(patient.chronic_conditions, item)

    def _append_source_ref(self, patient: Patient, source_type: str, source_path: str) -> None:
        existing = {(ref.source_type, ref.source_path) for ref in patient.source_refs}
        if (source_type, source_path) not in existing:
            patient.source_refs.append(SourceReference(source_type=source_type, source_path=source_path))

    def _append_unique(self, items: list[str], value: str) -> None:
        normalized_value = value.strip()
        if normalized_value and normalized_value not in items:
            items.append(normalized_value)

    def _looks_chronic(self, condition_name: str) -> bool:
        lowered = condition_name.lower()
        return any(token in lowered for token in ('diabetes', 'hypertension', 'kidney', 'migraine', 'pcos', 'asthma'))

    def _normalize_header_name(self, value: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', value.strip().lower()).strip('_')

    def _normalize_phone(self, value: str | None) -> str | None:
        if not value:
            return None
        digits = ''.join(ch for ch in value if ch.isdigit())
        return digits or None

    def _normalize_name(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()
        return normalized or None

    def _make_patient_id(self, name: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        return f'pat_{slug or "demo"}'

    def _make_doctor_id(self, name: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        return f'doc_{slug or "demo"}'

    def _normalize_specialty(self, specialty: str) -> str:
        normalized = specialty.strip()
        corrections = {
            'pulmonoly': 'Pulmonology',
            'general medicine': 'General Medicine',
            'orthopedic surgeon': 'Orthopedic Surgeon',
        }
        key = normalized.lower()
        corrected = corrections.get(key, normalized)
        return corrected.title() if corrected.islower() else corrected

    def _parse_date(self, value: str | None):
        if not value:
            return None
        value = value.strip()
        for fmt in ('%m/%d/%Y', '%m/%d/%Y %I:%M %p', '%Y-%m-%d'):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None
