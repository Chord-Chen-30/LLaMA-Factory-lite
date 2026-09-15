"""LLM-based <think> rephraser used by level 2a / 3a merges.

Sends the concatenated <think> bodies through MERGE_THINK_PROMPT (defined in
prompt.py) and returns the rephrased coherent reasoning passage.

Design mirrors call_openrouter.py:
- Try NEW_API (cheap) first, fall back to OPENROUTER (primary).
- Same env vars: NEW_API / NEW_API_URL / NEW_API_MODEL,
  OPENROUTER_API_KEY / OPENROUTER_MODEL.

Differences from call_openrouter.py:
- No sys_prompt / agent system message — we just send the merge prompt as user.
- Adds a disk-backed cache keyed by hash of input, so re-runs and 2a/3a sharing
  the same merge group don't pay twice.
"""

import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
from typing import Optional

import requests
from dotenv import load_dotenv

# Load MERGE_THINK_PROMPT from the prompt.py that lives next to THIS file —
# avoids collision with agent_infer/src/prompt.py which other modules add to sys.path.
_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.py")
_spec = importlib.util.spec_from_file_location("data_optim_prompt", _PROMPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MERGE_THINK_PROMPT = _mod.MERGE_THINK_PROMPT


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
NEW_API_KEY = os.getenv("NEW_API")
NEW_API_URL = os.getenv("NEW_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
NEW_API_MODEL = os.getenv("NEW_API_MODEL", "gpt-5.4")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.4")

logger = logging.getLogger(__name__)

# Disk-backed cache: hash -> rephrased text. Avoids re-paying for repeated calls
# (e.g. running level 2a and 3a back-to-back over overlapping merge groups, or
# resuming after a crash).
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_think_cache.json")
_cache = None
_cache_lock = threading.Lock()


def _cache_get(key: str) -> Optional[str]:
    global _cache
    with _cache_lock:
        if _cache is None:
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                _cache = {}
        return _cache.get(key)


def _cache_put(key: str, value: str) -> None:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = {}
        _cache[key] = value
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_cache, f, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"merge-think cache write failed: {e}")


def _post_chat(url: str, key: str, data: dict, max_retries: int, label: str) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=data, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError(f"{label} failed with no exception")


def rephrase_thinks(concat_thinks: str, max_retries_cheap: int = 5, max_retries_primary: int = 12) -> str:
    """Rephrase a `;\\n`-joined chain of thoughts into one coherent passage.

    Returns the cleaned rephrased text. On total LLM failure, raises — the caller
    is responsible for a graceful fallback (e.g., returning the original concat).
    """
    key = hashlib.sha256(concat_thinks.encode("utf-8")).hexdigest()
    cached = _cache_get(key)
    if cached is not None:
        return cached

    prompt = MERGE_THINK_PROMPT.format(CONCATED_THINK=concat_thinks)
    messages = [{"role": "user", "content": prompt}]

    last_err = None
    if NEW_API_KEY:
        data = {"model": NEW_API_MODEL, "stream": False, "messages": messages}
        try:
            resp = _post_chat(NEW_API_URL, NEW_API_KEY, data, max_retries_cheap, "NEW_API")
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                content = content.strip()
                _cache_put(key, content)
                return content
        except Exception as e:
            last_err = e
            logger.warning(f"NEW_API rephrase failed after {max_retries_cheap} retries: {e}; falling back")

    if OPENROUTER_KEY:
        data = {"model": OPENROUTER_MODEL, "stream": False, "messages": messages}
        try:
            resp = _post_chat(OPENROUTER_URL, OPENROUTER_KEY, data, max_retries_primary, "OPENROUTER")
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                content = content.strip()
                _cache_put(key, content)
                return content
        except Exception as e:
            last_err = e
            logger.error(f"OPENROUTER rephrase failed after {max_retries_primary} retries: {e}")

    raise RuntimeError(f"rephrase_thinks: both endpoints failed; last_err={last_err}")


if __name__ == "__main__":
    # smoke test
    logging.basicConfig(level=logging.INFO)
    sample = (
        "I need to identify the painting in the image, so I'll do a reverse image search;\n"
        "The image was identified as Vermeer's Girl with a Pearl Earring. Now I need its current location;\n"
        "I have all the facts I need: the painting is in the Mauritshuis museum in The Hague."
    )
    out = rephrase_thinks(sample)
    print(f"Input chars: {len(sample)}")
    print(f"Output chars: {len(out)}")
    print(f"\n--- Output ---\n{out}")
