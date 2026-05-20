from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT_DIR / "samples"
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DATABASE_PATH = DATA_DIR / "healthcare_assistant.db"
ENV_FILE = ROOT_DIR / ".env"


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == name:
                cleaned = raw_value.strip().strip('"').strip("'")
                if cleaned:
                    os.environ[name] = cleaned
                    return cleaned
    return default
