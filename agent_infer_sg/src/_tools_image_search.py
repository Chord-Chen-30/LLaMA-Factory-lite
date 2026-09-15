"""Reverse image search via Google Lens (Serper), with imgbb upload + disk caches.

Extracted from ``tools.py`` to keep that file focused; the public symbol
``image_search`` is re-exported from ``tools`` so existing callers
(``from tools import image_search``) keep working unchanged.

Endpoint strategy (per call, after cache miss):
  1. Primary: MiroMind Serper proxy ``${SERPER_BASE_URL}/search/lens``
     with the single ``SERPER_API_KEY`` (same key used by web search).
  2. Fallback (on any HTTP / network error): direct
     ``${SERPER_LENS_URL}`` (default ``https://google.serper.dev/lens``),
     iterating through the comma-separated ``SERPER_LENS_API_KEY`` list.

Caches (all next to this file):
  - image_url_map.json    : local path  -> imgbb URL  (avoids re-uploading)
  - image_search_cache.json : query URL -> raw Serper Lens response
                              (avoids re-calling Serper for the same URL)
    Note: cache key is the query URL only; ``num``/``page`` arguments do not
    participate in the key. Default callers (num=10, page=1) hit the cache
    transparently; non-default callers accept slight staleness.

Concurrency:
  - image_url_map.json     : ``fcntl.flock`` across processes
  - image_search_cache.json: in-process ``threading.Lock`` only; multi-process
    races may overwrite one entry — acceptable (worst case: one redundant API hit).
"""
import fcntl
import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

# Same .env as tools.py (lives one level up in the same dir).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from utils import image_to_base64

logger = logging.getLogger(__name__)

# =============================================================================
# Keys / endpoints
# =============================================================================
# Primary Lens endpoint: MiroMind Serper proxy. Single shared key (same one
# used by web search). Try this first.
SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_LENS_PROXY_URL = f"{SERPER_BASE_URL.rstrip('/')}/search/lens"

# Legacy fallback: direct google.serper.dev/lens with per-key fallback list.
# Used only if the proxy call above fails.
SERPER_LENS_URL = os.getenv("SERPER_LENS_URL", "https://google.serper.dev/lens")
SERPER_LENS_API_KEYS = [k.strip() for k in os.getenv("SERPER_LENS_API_KEY", "").split(",") if k.strip()]

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")
IMGBB_UPLOAD_URL = os.getenv("IMGBB_UPLOAD_URL", "https://api.imgbb.com/1/upload")
IMGBB_EXPIRATION = int(os.getenv("IMGBB_EXPIRATION", "0") or "0")

IMAGE_URL_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_url_map.json")
LENS_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_search_cache.json")


# =============================================================================
# imgbb upload + local-path → public URL cache
# =============================================================================
# Local path → public URL cache. Persisted as JSON in src/. Concurrent workers
# coordinate via a sidecar lockfile (fcntl.flock). Upload happens outside the
# lock — two workers uploading the same new image race to a duplicate upload
# (accepted cost; imgbb quota is generous) and the last writer wins the map
# entry. Subsequent lookups hit the cache.
def _map_lock(fn):
    def _wrapped(*args, **kwargs):
        os.makedirs(os.path.dirname(IMAGE_URL_MAP_PATH), exist_ok=True)
        lock_path = IMAGE_URL_MAP_PATH + ".lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fn(*args, **kwargs)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return _wrapped


def _load_image_url_map() -> dict:
    try:
        with open(IMAGE_URL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@_map_lock
def _insert_image_url(local_abspath: str, public_url: str) -> str:
    m = _load_image_url_map()
    existing = m.get(local_abspath)
    if existing:
        return existing
    m[local_abspath] = public_url
    d = os.path.dirname(IMAGE_URL_MAP_PATH)
    with tempfile.NamedTemporaryFile("w", dir=d, prefix=".imgmap.", suffix=".tmp",
                                     delete=False, encoding="utf-8") as tmp:
        json.dump(m, tmp, indent=2, sort_keys=True, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.rename(tmp_name, IMAGE_URL_MAP_PATH)
    return public_url


def _upload_imgbb(path: str) -> str:
    if not IMGBB_API_KEY:
        raise RuntimeError("IMGBB_API_KEY not set")
    data_url = image_to_base64(path)
    b64 = data_url.split(",", 1)[1]  # strip "data:<mime>;base64," prefix
    resp = requests.post(
        IMGBB_UPLOAD_URL,
        params={"key": IMGBB_API_KEY, "expiration": IMGBB_EXPIRATION},
        data={"image": b64},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"imgbb upload failed: {data}")
    return data["data"]["url"]


def _get_public_url(path: str) -> str:
    abspath = os.path.abspath(path.replace("file://", ""))
    cached = _load_image_url_map().get(abspath)
    if cached:
        return cached
    if not os.path.exists(abspath):
        raise FileNotFoundError(f"image not found: {abspath}")
    public_url = _upload_imgbb(abspath)
    return _insert_image_url(abspath, public_url)


# =============================================================================
# Lens response cache: query_url -> raw Serper JSON
# =============================================================================
_lens_cache = None
_lens_cache_lock = threading.Lock()


def _lens_cache_get(url: str):
    global _lens_cache
    with _lens_cache_lock:
        if _lens_cache is None:
            try:
                with open(LENS_CACHE_PATH, "r", encoding="utf-8") as f:
                    _lens_cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                _lens_cache = {}
        return _lens_cache.get(url)


def _lens_cache_put(url: str, data: dict) -> None:
    global _lens_cache
    with _lens_cache_lock:
        if _lens_cache is None:
            _lens_cache = {}
        _lens_cache[url] = data
        try:
            with open(LENS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_lens_cache, f, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"Lens cache write failed: {e}")


# =============================================================================
# Result formatting + single-image dispatch
# =============================================================================
def _format_lens_results(data: dict, top_n: int = 10) -> str:
    organic = data.get("organic") or []
    if not organic:
        return "No image search results found."
    parts = []
    for item in organic[:top_n]:
        parts.append(
            f"Title: {item.get('title', '')}\n"
            f"Source: {item.get('source', '')}\n"
            f"URL: {item.get('link', '')}\n"
            f"Image: {item.get('imageUrl', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _call_lens_proxy(query_url: str, num: int, page: int) -> dict:
    """Primary path: MiroMind Serper proxy with the single SERPER_API_KEY."""
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not configured")
    resp = requests.post(
        SERPER_LENS_PROXY_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"url": query_url, "num": num, "page": page},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _call_lens_legacy(query_url: str) -> dict:
    """Fallback path: direct google.serper.dev/lens, trying each key in order."""
    keys = [k for k in SERPER_LENS_API_KEYS if k]
    if not keys:
        raise RuntimeError("no SERPER_LENS_API_KEY configured")
    last_err = None
    for idx, key in enumerate(keys):
        try:
            resp = requests.post(
                SERPER_LENS_URL,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"url": query_url},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            logger.warning(
                f"Serper Lens key #{idx+1} failed: {e}; trying next"
                if idx + 1 < len(keys)
                else f"Serper Lens key #{idx+1} failed: {e}; no more keys"
            )
    raise last_err


def _image_search_single(image: str, info_to_extract: str = "",
                         num: int = 10, page: int = 1) -> str:
    try:
        # 1. Resolve to a Serper-reachable URL (imgbb upload for local files).
        if isinstance(image, str) and re.match(r"^https?://", image):
            query_url = image
        else:
            query_url = _get_public_url(image)

        # 2. Cache lookup. Key is query_url only, so default-`num`/`page`
        #    callers always hit; non-default callers accept slight staleness.
        data = _lens_cache_get(query_url)
        if data is not None:
            logger.info(f"[lens cache HIT] {query_url}")
        else:
            # 3a. Primary: MiroMind proxy with single SERPER_API_KEY.
            try:
                data = _call_lens_proxy(query_url, num=num, page=page)
            except Exception as e:
                logger.warning(f"Serper proxy failed ({e}); falling back to direct Lens API")
                # 3b. Fallback: direct google.serper.dev/lens with key list.
                data = _call_lens_legacy(query_url)
            _lens_cache_put(query_url, data)

        # 4. Format + optional summarizer.
        results = _format_lens_results(data)
        if info_to_extract and results and not results.startswith("No image"):
            # Late import: avoids a top-level circular dependency with tools.py
            # (tools.py imports image_search from this module at load time).
            from tools import _summarize_content
            return _summarize_content(results, info_to_extract)
        return results
    except Exception as e:
        logger.error(f"Image search error for {image}: {e}")
        return f"Image search error: {e}"


# =============================================================================
# Public entry point
# =============================================================================
def image_search(image, info_to_extract: str = "",
                 num: int = 10, page: int = 1) -> str:
    """Reverse image search via Google Lens (Serper).

    Accepts a local path, an ``http(s)://`` URL, or a list of either. Local
    paths are first uploaded to imgbb (cached in ``image_url_map.json``); the
    resulting imgbb URL is then queried against Serper's Lens endpoint, with
    the Serper response cached in ``image_search_cache.json``.

    Lens calls go through the MiroMind proxy first (single SERPER_API_KEY),
    falling back to the direct google.serper.dev/lens endpoint (with the
    SERPER_LENS_API_KEY fallback list) on any error.

    ``num``/``page`` are forwarded to the proxy endpoint only; the legacy
    fallback ignores them.
    """
    if isinstance(image, list):
        if isinstance(info_to_extract, list):
            extracts = info_to_extract + [""] * max(0, len(image) - len(info_to_extract))
        else:
            extracts = [info_to_extract] * len(image)
        results = [None] * len(image)
        with ThreadPoolExecutor(max_workers=min(len(image), 8)) as ex:
            futs = {
                ex.submit(_image_search_single, im, extracts[i], num, page): i
                for i, im in enumerate(image)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = f"Image search error: {e}"
        return "\n\n".join(
            f"=== Image [{i+1}/{len(image)}]: {im} ===\n{r}"
            for i, (im, r) in enumerate(zip(image, results))
        )
    return _image_search_single(image, info_to_extract, num=num, page=page)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("=== image_search (local) ===")
    print(image_search(
        "./agent_infer_sg/data/livevqa_files/0a5o7ZLh8b.jpeg"
    )[:800])
