from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

from app.repositories.sample_data_repository import SampleDataRepository
from app.utils.config import CHROMA_DIR, get_env


@dataclass
class ChunkedDocument:
    chunk_id: str
    document: str
    metadata: dict[str, object]


class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by the OpenAI embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key or get_env("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to the environment or .env file.")
        self.client = OpenAI(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        kwargs: dict[str, object] = {
            "model": self.model,
            "input": list(input),
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


class ChromaIngestionService:
    """Ingests normalized sample data into local ChromaDB collections."""

    def __init__(
        self,
        repository: SampleDataRepository | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int | None = None,
    ) -> None:
        self.repository = repository or SampleDataRepository()
        self.embedding_function = OpenAIEmbeddingFunction(
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    def ingest_all(self, reset: bool = True) -> dict[str, int]:
        if reset:
            self._reset_collection("patient_records_structured")
            self._reset_collection("patient_reports_unstructured")
            self._reset_collection("patient_memory_summaries")

        structured_count = self._ingest_structured_records()
        unstructured_count = self._ingest_unstructured_documents()
        memory_count = self._ingest_patient_memories()
        return {
            "patient_records_structured": structured_count,
            "patient_reports_unstructured": unstructured_count,
            "patient_memory_summaries": memory_count,
        }

    def query(self, collection_name: str, query_text: str, n_results: int = 3) -> dict[str, object]:
        collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_function)
        return collection.query(query_texts=[query_text], n_results=n_results)

    def _ingest_structured_records(self) -> int:
        collection = self.client.get_or_create_collection(
            name="patient_records_structured",
            embedding_function=self.embedding_function,
            metadata={"description": "Structured patient record content", "embedding_model": self.embedding_function.model},
        )
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, object]] = []
        for record in self.repository.records:
            ids.append(record.record_id)
            documents.append(self._record_to_text(record))
            metadatas.append(
                {
                    "patient_id": record.patient_id,
                    "entry_type": record.entry_type.value,
                    "visit_date": record.visit_date.isoformat() if record.visit_date else "",
                    "source_path": record.source_path,
                    "source_type": record.source_type,
                }
            )
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def _ingest_unstructured_documents(self) -> int:
        collection = self.client.get_or_create_collection(
            name="patient_reports_unstructured",
            embedding_function=self.embedding_function,
            metadata={"description": "Chunked raw patient PDF content", "embedding_model": self.embedding_function.model},
        )
        chunks: list[ChunkedDocument] = []
        for raw_pdf in self.repository.raw_pdf_documents:
            patient = self.repository.get_patient_by_name(self._extract_patient_name(raw_pdf.extracted_text))
            patient_id = patient.patient_id if patient else None
            for index, chunk in enumerate(self._chunk_text(raw_pdf.extracted_text), start=1):
                chunks.append(
                    ChunkedDocument(
                        chunk_id=f"{raw_pdf.source_file}::chunk_{index:03d}",
                        document=chunk,
                        metadata={
                            "patient_id": patient_id or "",
                            "document_type": raw_pdf.document_type,
                            "source_file": raw_pdf.source_file,
                            "source_path": raw_pdf.source_path,
                        },
                    )
                )
        if chunks:
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.document for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )
        return len(chunks)

    def _ingest_patient_memories(self) -> int:
        collection = self.client.get_or_create_collection(
            name="patient_memory_summaries",
            embedding_function=self.embedding_function,
            metadata={"description": "Derived patient memory summaries", "embedding_model": self.embedding_function.model},
        )
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, object]] = []
        for patient in self.repository.patients:
            records = self.repository.get_records_for_patient(patient.patient_id)
            diagnoses = self.repository.get_diagnoses_for_patient(patient.patient_id)
            summary = self._build_memory_summary(patient.full_name, records, diagnoses)
            ids.append(f"mem::{patient.patient_id}")
            documents.append(summary)
            metadatas.append(
                {
                    "patient_id": patient.patient_id,
                    "full_name": patient.full_name,
                    "source_count": len(patient.source_refs),
                }
            )
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def _build_memory_summary(self, patient_name: str, records: Iterable, diagnoses: Iterable) -> str:
        diagnosis_names = [diagnosis.name for diagnosis in diagnoses if diagnosis.name]
        plan_lines = [record.plan for record in records if record.plan]
        parts = [f"Patient: {patient_name}."]
        if diagnosis_names:
            parts.append("Diagnoses: " + "; ".join(diagnosis_names) + ".")
        if plan_lines:
            parts.append("Prior plans: " + " ".join(plan_lines))
        return " ".join(parts)

    def _record_to_text(self, record) -> str:
        parts = [record.title]
        for value in (record.subjective, record.objective, record.assessment, record.plan):
            if value:
                parts.append(value)
        return "\n".join(parts)

    def _chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            chunks.append(normalized[start:end])
            if end >= len(normalized):
                break
            start = max(0, end - overlap)
        return chunks

    def _extract_patient_name(self, text: str) -> str:
        match = re.search(r"Patient:\s*(.+)", text)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0] if lines else ""

    def _reset_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
