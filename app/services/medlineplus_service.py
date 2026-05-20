from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

from app.schemas.contracts import MedlinePlusResult, SearchMedlinePlusTopicsOutput

_MEDLINEPLUS_WEBSEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
_MEDLINEPLUS_CONNECT_URL = "https://connect.medlineplus.gov/service"
_CODE_SYSTEM_OIDS = {
    "ICD-10": "2.16.840.1.113883.6.90",
    "ICD-9-CM": "2.16.840.1.113883.6.103",
    "SNOMED CT": "2.16.840.1.113883.6.96",
    "RXNORM": "2.16.840.1.113883.6.88",
    "LOINC": "2.16.840.1.113883.6.1",
}
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class MedlinePlusService:
    """Adapter for MedlinePlus Web Service and MedlinePlus Connect."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def search_topics(self, query: str, limit: int = 3) -> SearchMedlinePlusTopicsOutput:
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": str(limit),
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(_MEDLINEPLUS_WEBSEARCH_URL, params=params)
            response.raise_for_status()
        return SearchMedlinePlusTopicsOutput(results=self._parse_web_search_xml(response.text, limit=limit))

    def search_from_codes(self, diagnoses: list[str], limit: int = 3) -> SearchMedlinePlusTopicsOutput:
        results: list[MedlinePlusResult] = []
        for diagnosis in diagnoses:
            code_system, code, title = self._extract_code_details(diagnosis)
            if not code_system or not code:
                continue
            oid = _CODE_SYSTEM_OIDS.get(code_system)
            if not oid:
                continue
            params = {
                "mainSearchCriteria.v.cs": oid,
                "mainSearchCriteria.v.c": code,
                "mainSearchCriteria.v.dn": title or diagnosis,
                "informationRecipient.languageCode.c": "en",
                "knowledgeResponseType": "application/xml",
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(_MEDLINEPLUS_CONNECT_URL, params=params)
                response.raise_for_status()
            results.extend(self._parse_connect_atom(response.text, limit=limit))
            if len(results) >= limit:
                break
        deduped: list[MedlinePlusResult] = []
        seen = set()
        for item in results:
            key = item.url
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return SearchMedlinePlusTopicsOutput(results=deduped)

    def _parse_web_search_xml(self, xml_text: str, limit: int) -> list[MedlinePlusResult]:
        root = ET.fromstring(xml_text)
        results: list[MedlinePlusResult] = []
        for document in root.findall(".//document")[:limit]:
            content_map: dict[str, str] = {}
            for content in document.findall("content"):
                name = (content.attrib.get("name") or "").lower()
                value = "".join(content.itertext()).strip()
                if name and value:
                    content_map[name] = value
            title = content_map.get("title") or content_map.get("fulltitle") or content_map.get("groupname") or "MedlinePlus Topic"
            url = document.attrib.get("url") or content_map.get("url") or content_map.get("link") or "https://medlineplus.gov/"
            snippet = content_map.get("fullsummary") or content_map.get("snippet") or content_map.get("groupname")
            results.append(
                MedlinePlusResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_group="medlineplus_web",
                )
            )
        return results

    def _parse_connect_atom(self, xml_text: str, limit: int) -> list[MedlinePlusResult]:
        root = ET.fromstring(xml_text)
        results: list[MedlinePlusResult] = []
        for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
            title = (entry.findtext("atom:title", default="MedlinePlus Connect", namespaces=_ATOM_NS) or "MedlinePlus Connect").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
            link = entry.find("atom:link", _ATOM_NS)
            url = link.attrib.get("href") if link is not None else "https://medlineplus.gov/"
            results.append(
                MedlinePlusResult(
                    title=title,
                    url=url,
                    snippet=summary,
                    source_group="medlineplus_connect",
                )
            )
        return results

    def _extract_code_details(self, diagnosis_text: str) -> tuple[str | None, str | None, str | None]:
        icd10_match = re.search(r"(.+?)\(([A-Z]\d+(?:\.\d+)?)\)", diagnosis_text)
        if icd10_match:
            return "ICD-10", icd10_match.group(2), icd10_match.group(1).replace("Diagnosis:", "").strip()
        icd10_bracket = re.search(r"(.+?)\[ICD-10:\s*([A-Z]\d+(?:\.\d+)?)\]", diagnosis_text)
        if icd10_bracket:
            return "ICD-10", icd10_bracket.group(2), icd10_bracket.group(1).replace("DIAGNOSIS:", "").strip()
        return None, None, None
