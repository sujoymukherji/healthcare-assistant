from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.utils.config import ROOT_DIR


PROMPTS_DIR = ROOT_DIR / "app" / "prompts"


@lru_cache(maxsize=128)
def load_prompt(relative_path: str) -> str:
    prompt_path = PROMPTS_DIR / relative_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()
