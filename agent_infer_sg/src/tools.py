"""Tool implementations for agent_infer_sg.

Changes vs. source:
  - API keys loaded from local .env.
  - E2B sandbox dropped; run_python uses a local subprocess.
  - Summary-LLM path kept but disabled by default (SUMMARY_LLM_BASE_URL unset) —
    visit_url returns truncated raw content unless the caller configures one.
"""
import os
import re
import json
import time
import fcntl
import logging
import subprocess
import sys
import tempfile
import threading
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from utils import image_to_base64

logger = logging.getLogger(__name__)

# =============================================================================
# Keys / endpoints
# =============================================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
SERPER_SEARCH_URL = f"{SERPER_BASE_URL}/search"

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_BASE_URL = os.getenv("JINA_BASE_URL", "https://r.jina.ai")

SUMMARY_LLM_BASE_URL = os.getenv("SUMMARY_LLM_BASE_URL", "")
SUMMARY_LLM_MODEL_NAME = os.getenv("SUMMARY_LLM_MODEL_NAME", "")
SUMMARY_LLM_API_KEY = os.getenv("SUMMARY_LLM_API_KEY", "")

CODE_EXEC_TIMEOUT = 600  # 10 min — plenty for sympy/scipy on benchmark items


# =============================================================================
# Summary LLM
# =============================================================================
EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def _summarize_content(content: str, info_to_extract: str) -> str:
    if not SUMMARY_LLM_BASE_URL or not SUMMARY_LLM_MODEL_NAME:
        if len(content) > 20000:
            content = content[:20000] + "\n\n[Content truncated...]"
        return content

    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    model = SUMMARY_LLM_MODEL_NAME
    if "gpt-5" in model.lower() or "gpt5" in model.lower():
        payload = {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "minimal",
        }
    else:
        payload = {
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
        }

    headers = {"Content-Type": "application/json"}
    if SUMMARY_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {SUMMARY_LLM_API_KEY}"

    truncate_step = 40960
    for attempt in range(4):
        try:
            resp = requests.post(SUMMARY_LLM_BASE_URL, headers=headers, json=payload, timeout=300)
            if ("maximum context length" in resp.text or "longer than the model's context length" in resp.text):
                chars_to_remove = truncate_step * (attempt + 1)
                if chars_to_remove < len(content):
                    truncated = content[:-chars_to_remove] + "[...truncated]"
                    payload["messages"][0]["content"] = EXTRACT_INFO_PROMPT.format(info_to_extract, truncated)
                    continue
                else:
                    break
            resp.raise_for_status()
            summary = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if summary:
                return summary
            break
        except Exception as e:
            logger.error(f"[Summary LLM] attempt {attempt+1}: {e}")
            if attempt < 3:
                time.sleep(attempt + 1)
                continue
            break

    if len(content) > 20000:
        content = content[:20000] + "\n\n[Content truncated...]"
    return content


# =============================================================================
# Search
# =============================================================================
def _search_single(query: str, num: int = 10) -> str:
    try:
        resp = requests.post(
            SERPER_SEARCH_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic", []):
            results.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('link', '')}\n"
                f"Snippet: {item.get('snippet', '')}"
            )
        kg = data.get("knowledgeGraph", {})
        if kg:
            kg_text = f"Knowledge Graph: {kg.get('title', '')} - {kg.get('description', '')}"
            for k, v in (kg.get("attributes") or {}).items():
                kg_text += f"\n  {k}: {v}"
            results.insert(0, kg_text)
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"


def search(query, num: int = 10) -> str:
    """Google search via Serper. Accepts str or list[str]."""
    if isinstance(query, list):
        results = [None] * len(query)
        with ThreadPoolExecutor(max_workers=min(len(query), 8)) as ex:
            futs = {ex.submit(_search_single, q, num): i for i, q in enumerate(query)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = f"Search error: {e}"
        return "\n\n".join(f"=== Search [{i+1}/{len(query)}]: {q} ===\n{r}" for i, (q, r) in enumerate(zip(query, results)))
    return _search_single(query, num)


# =============================================================================
# URL fetching — smart routing for paywalled/anti-bot domains
# =============================================================================
_PAYWALL_DOMAINS = {
    "www.sciencedirect.com", "linkinghub.elsevier.com",
    "onlinelibrary.wiley.com", "chemistry-europe.onlinelibrary.wiley.com",
    "advanced.onlinelibrary.wiley.com", "analyticalsciencejournals.onlinelibrary.wiley.com",
    "nph.onlinelibrary.wiley.com", "faseb.onlinelibrary.wiley.com",
    "link.aps.org", "journals.aps.org",
    "www.cell.com", "academic.oup.com", "www.tandfonline.com",
    "pubs.rsc.org", "www.rsc.org", "www.science.org",
    "www.jneurosci.org", "ashpublications.org",
    "pubs.aip.org", "journals.sagepub.com", "pubs.acs.org",
}


def _jina_fetch(url: str) -> str:
    resp = requests.post(
        JINA_BASE_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/plain",
        },
        json={"url": url},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


def _is_garbage_content(text: str) -> bool:
    if not text:
        return True
    low = text[:2000].lower()
    signals = [
        "security verification", "captcha", "cloudflare",
        "access denied", "please verify", "robot",
        "enable javascript", "browser check",
        "cookies are required", "please enable cookies",
        "sign in to access", "institutional login",
        "subscribe to read", "purchase this article",
    ]
    return any(s in low for s in signals)


def _extract_pmcid(url: str) -> str:
    m = re.search(r"(PMC\d+)", url)
    return m.group(1) if m else ""


def _extract_doi(url: str) -> str:
    m = re.search(r"doi\.org/(10\.\d{4,}/[^\s]+)", url)
    if m:
        return m.group(1).rstrip("/")
    m = re.search(r"/(10\.\d{4,}/[^\s?#]+)", url)
    if m:
        return m.group(1).rstrip("/")
    return ""


def _fetch_pmc_fulltext(pmcid: str) -> str:
    try:
        resp = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode",
            timeout=30,
        )
        if resp.status_code != 200 or len(resp.text) < 500:
            return ""
        if resp.text.strip().startswith("[Error]") or "<html" in resp.text[:100].lower():
            return ""
        data = resp.json()
        items = data if isinstance(data, list) else [data]
        parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for doc in item.get("documents", []):
                for passage in doc.get("passages", []):
                    t = passage.get("text", "")
                    if t:
                        parts.append(t)
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[PMC] {pmcid}: {e}")
        return ""


def _pubmed_to_pmc(url: str) -> str:
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if not m:
        return ""
    pmid = m.group(1)
    try:
        resp = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json",
            timeout=15,
        )
        if resp.status_code != 200:
            return ""
        for ls in resp.json().get("linksets", []):
            for ldb in ls.get("linksetdbs", []):
                if ldb.get("dbto") == "pmc" and ldb.get("links"):
                    return f"PMC{ldb['links'][0]}"
    except Exception as e:
        logger.warning(f"[PubMed->PMC] {pmid}: {e}")
    return ""


def _fetch_unpaywall_oa_url(doi: str) -> str:
    if not doi:
        return ""
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={os.getenv('UNPAYWALL_EMAIL', 'unpaywall@example.com')}",
            timeout=15,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        best = data.get("best_oa_location") or {}
        pdf = best.get("url_for_pdf") or best.get("url") or ""
        if pdf:
            return pdf
        for loc in data.get("oa_locations", []):
            pdf = loc.get("url_for_pdf") or loc.get("url") or ""
            if pdf:
                return pdf
    except Exception as e:
        logger.warning(f"[Unpaywall] {doi}: {e}")
    return ""


def _biorxiv_to_pdf(url: str) -> str:
    if not ("biorxiv.org" in url or "medrxiv.org" in url):
        return ""
    clean = url.split("?")[0].split("#")[0].rstrip("/")
    if clean.endswith(".pdf"):
        return url
    if "/content/" in clean:
        return clean + ".full.pdf"
    return ""


def _visit_url_single(url: str, info_to_extract: str = "") -> str:
    domain = urlparse(url).netloc
    raw_text = ""
    try:
        if "pmc.ncbi.nlm.nih.gov" in domain:
            pmcid = _extract_pmcid(url)
            if pmcid:
                raw_text = _fetch_pmc_fulltext(pmcid)
            if not raw_text:
                raw_text = _jina_fetch(url)

        elif "pubmed.ncbi.nlm.nih.gov" in domain:
            pmcid = _pubmed_to_pmc(url)
            if pmcid:
                raw_text = _fetch_pmc_fulltext(pmcid)
            if not raw_text:
                raw_text = _jina_fetch(url)

        elif "biorxiv.org" in domain or "medrxiv.org" in domain:
            pdf_url = _biorxiv_to_pdf(url)
            if pdf_url:
                try:
                    raw_text = _jina_fetch(pdf_url)
                except Exception:
                    pass
            if not raw_text or len(raw_text) < 500:
                raw_text = _jina_fetch(url)

        elif domain in _PAYWALL_DOMAINS:
            doi = _extract_doi(url)
            oa_url = _fetch_unpaywall_oa_url(doi) if doi else ""
            if oa_url:
                try:
                    raw_text = _jina_fetch(oa_url)
                except Exception:
                    pass
            if not raw_text or len(raw_text) < 1000:
                try:
                    raw_text = _jina_fetch(url)
                except Exception:
                    pass

        else:
            raw_text = _jina_fetch(url)

        if _is_garbage_content(raw_text):
            doi = _extract_doi(url) if not ("pmc.ncbi" in domain or "pubmed.ncbi" in domain) else ""
            oa_url = _fetch_unpaywall_oa_url(doi) if doi else ""
            if oa_url:
                try:
                    alt = _jina_fetch(oa_url)
                    if not _is_garbage_content(alt) and len(alt) > len(raw_text):
                        raw_text = alt
                except Exception:
                    pass

        if not raw_text:
            return f"Could not fetch content from {url}"

        if _is_garbage_content(raw_text):
            return (
                f"[ACCESS BLOCKED] The page at {url} is behind a paywall or anti-bot protection. "
                f"Try an open-access version on arxiv.org, PMC, or another open repository."
            )

        if info_to_extract:
            return _summarize_content(raw_text, info_to_extract)

        if len(raw_text) > 20000:
            raw_text = raw_text[:20000] + "\n\n[Content truncated...]"
        return raw_text

    except Exception as e:
        logger.error(f"Visit URL error for {url}: {e}")
        return f"Visit URL error: {e}"


def visit_url(url, info_to_extract: str = "") -> str:
    """Fetch page content. Accepts str or list[str]."""
    if isinstance(url, list):
        if isinstance(info_to_extract, list):
            extracts = info_to_extract + [""] * max(0, len(url) - len(info_to_extract))
        else:
            extracts = [info_to_extract] * len(url)
        results = [None] * len(url)
        with ThreadPoolExecutor(max_workers=min(len(url), 8)) as ex:
            futs = {ex.submit(_visit_url_single, u, extracts[i]): i for i, u in enumerate(url)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = f"Visit URL error: {e}"
        return "\n\n".join(f"=== URL [{i+1}/{len(url)}]: {u} ===\n{r}" for i, (u, r) in enumerate(zip(url, results)))
    return _visit_url_single(url, info_to_extract)


# =============================================================================
# Reverse image search — implementation lives in _tools_image_search.py.
# Re-exported here so existing callers (`from tools import image_search`) work.
# =============================================================================
from _tools_image_search import image_search  # noqa: E402,F401


# =============================================================================
# Python exec
# =============================================================================
def run_python(code: str, timeout: int = None) -> str:
    """Execute LLM-generated Python in a sandboxed temp cwd.

    Any files the code writes via relative paths (e.g. cv2.imwrite("foo.jpg"))
    land inside the temp dir and get deleted when this function returns, so they
    don't leak into the inference process's working directory.
    """
    if timeout is None:
        timeout = CODE_EXEC_TIMEOUT
    import shutil
    workdir = tempfile.mkdtemp(prefix="run_python_")
    script_path = os.path.join(workdir, "main.py")
    try:
        with open(script_path, "w") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                errors="replace",
                cwd=workdir,
            )
            parts = []
            if result.stdout:
                parts.append(result.stdout.strip())
            if result.stderr:
                parts.append(f"[stderr] {result.stderr.strip()}")
            out = "\n".join(parts).strip()
            return out if out else "(No output)"
        except subprocess.TimeoutExpired:
            return (
                f"Execution timed out after {timeout}s. Do not retry the same code — "
                "simplify the approach (numerical instead of symbolic, reduce dimensionality, etc.)."
            )
        except Exception as e:
            return f"Python execution error: {e}"
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("=== search ===")
    print(search("Who is Chord Chen?", num=3)[:500])
    print("\n=== visit_url ===")
    print(visit_url("https://chord-chen-30.github.io/")[:500])
    print("\n=== run_python ===")
    print(run_python("print(sum(range(100)))"))
    print("\n=== image_search (local) ===")
    print(image_search("./agent_infer_sg/data/livevqa_files/0a5o7ZLh8b.jpeg")[:800])
