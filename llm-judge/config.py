from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
    """Resolve a possibly-relative path against the llm-judge root."""
    p = Path(value)
    return str(p if p.is_absolute() else (BASE_DIR / p).resolve())


def _parse_weights(value: str) -> dict[str, float]:
    """'minor:1,major:5,critical:10' -> {'minor': 1.0, ...}"""
    out = {}
    for pair in value.split(","):
        k, v = pair.split(":")
        out[k.strip()] = float(v)
    return out


class Settings:
    # ---- Judge LLM (any OpenAI-compatible endpoint: Ollama, OpenAI, OpenRouter, vLLM...) ----
    PROVIDER = "anthropic" #NOTE swap to ollama for local models 
    #PROVIDER = "ollama" #NOTE swap to anthropic for claude models

    JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://localhost:11434/v1")
    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "ollama")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:14b-instruct")
    JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0"))
    JUDGE_SAMPLES = int(os.getenv("JUDGE_SAMPLES", "1"))
    JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "3"))
    JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "300"))
    # json_schema | json_mode | function_calling
    JUDGE_STRUCTURED = os.getenv("JUDGE_STRUCTURED", "json_schema")
    USE_GLOSSARY = os.getenv("JUDGE_USE_GLOSSARY", "1") == "1"

    # ---- Data ----
    TRANSLATE_IN_FILE = _resolve_path(os.getenv("TRANSLATE_IN_FILE", "../multilingual-assistant/results/translations/translate_in.json"))
    TRANSLATE_OUT_FILE = _resolve_path(os.getenv("TRANSLATE_OUT_FILE", "../multilingual-assistant/results/translations/translate_out.json"))
    FAILED_OUT = _resolve_path(os.getenv("FAILED_OUT_FILE", "../multilingual-assistant/results/translations/failed_out.json"))
    RESULTS_DIR = _resolve_path(os.getenv("RESULTS_DIR", "results"))

    # ---- Scoring ----
    MQM_WEIGHTS = _parse_weights(os.getenv("MQM_WEIGHTS", "minor:1,major:5,critical:10"))
    PASS_MAX_PENALTY = float(os.getenv("PASS_MAX_PENALTY", "5"))
    MIN_ITEMS_FOR_CORRELATION = int(os.getenv("MIN_ITEMS_FOR_CORRELATION", "5"))


settings = Settings()
