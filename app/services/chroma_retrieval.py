from __future__ import annotations

from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from app.schemas.domain import RetrievedEvidence
from app.services.chroma_ingestion import OpenAIEmbeddingFunction
from app.utils.config import CHROMA_DIR


class ChromaRetrievalService:
    """Retrieval helper for patient-context RAG over local Chroma collections."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int | None = None,
    ) -> None:
        self.embedding_function = OpenAIEmbeddingFunction(
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    def query_patient_records(self, patient_id: str, query_text: str, n_results: int = 3) -> list[RetrievedEvidence]:
        return self._query_collection(
            collection_name="patient_records_structured",
            query_text=query_text,
            n_results=n_results,
            where={"patient_id": patient_id},
            source_group="patient_history",
        )

    def query_patient_memory(self, patient_id: str, query_text: str, n_results: int = 2) -> list[RetrievedEvidence]:
        return self._query_collection(
            collection_name="patient_memory_summaries",
            query_text=query_text,
            n_results=n_results,
            where={"patient_id": patient_id},
            source_group="patient_memory",
        )

    def _query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int,
        where: dict[str, Any],
        source_group: str,
    ) -> list[RetrievedEvidence]:
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception:
            return []

        result = collection.query(query_texts=[query_text], n_results=n_results, where=where)
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        evidence: list[RetrievedEvidence] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            score = None if distance is None else 1.0 / (1.0 + float(distance))
            evidence.append(
                RetrievedEvidence(
                    evidence_id=f"{collection_name}::{chunk_id}",
                    source_group=source_group,
                    source_type=str(metadata.get("document_type") or metadata.get("entry_type") or collection_name),
                    source_label=str(metadata.get("source_file") or metadata.get("source_path") or chunk_id),
                    source_uri=str(metadata.get("source_path") or metadata.get("source_file") or ""),
                    patient_id=str(metadata.get("patient_id") or "") or None,
                    chunk_id=chunk_id,
                    text=docs[idx] if idx < len(docs) else "",
                    score=score,
                    metadata={str(k): v for k, v in metadata.items()},
                )
            )
        return evidence
