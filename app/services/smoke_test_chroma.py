from __future__ import annotations

import json

from app.services.chroma_ingestion import ChromaIngestionService


def main() -> None:
    service = ChromaIngestionService()
    counts = service.ingest_all(reset=True)
    sample_queries = {
        "patient_records_structured": "dry cough and mild fever",
        "patient_reports_unstructured": "type 2 diabetes increased thirst urination",
        "patient_memory_summaries": "hypertension routine follow up",
    }
    results = {"counts": counts, "queries": {}}
    for collection_name, query_text in sample_queries.items():
        query_result = service.query(collection_name, query_text, n_results=2)
        results["queries"][collection_name] = {
            "query": query_text,
            "ids": query_result.get("ids"),
            "documents": query_result.get("documents"),
            "metadatas": query_result.get("metadatas"),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
