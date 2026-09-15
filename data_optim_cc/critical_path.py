"""
Scheme B: Critical-Path Pruning.

Input:  ./data/toolcall_3more_ocr_code_6200_cz_cleaned_1216.jsonl
Output: jsonl of the same records, with `messages` replaced by the pruned trajectory and
        two extra fields:
          - `original_messages`: the untouched original for reference/debugging.
          - `cp_meta`: {"status", "removed": [...], "patched": {...}, "reason", "raw_llm"}.

The LLM is asked to identify middle rounds whose output is NOT actually used in the final
answer, and to provide minimal <think> rewrites for any kept round whose opening would
otherwise dangle (e.g. "let's try another link") after an adjacent deletion.

Round numbering (same convention as one_time_prompting.py):
  Round 1   ↔ messages[0]              (user question)
  Round r   ↔ messages[2r-3] (asst)    + messages[2r-2] (user/tool_response)   for 2 <= r <= N-1
  Round N   ↔ messages[-1]             (final assistant answer)
The LLM may only delete rounds 2..N-1.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_optim"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent_infer" / "src"))

from call_openrouter import call_openrouter  # noqa: E402
from utils import loads_llm_json  # noqa: E402
from prompt_cp import USER_PROMPT_TEMPLATE  # noqa: E402


CLEAN_KEEP = {"messages", "images", "system"}


def clean_for_training(src: str, dst: str) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            d2 = {k: v for k, v in d.items() if k in CLEAN_KEEP}
            fout.write(json.dumps(d2, ensure_ascii=False) + "\n")
            n += 1
    return n


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Grounding-check token extractors: capitalized multi-word phrases, 4-digit years, percentages.
_CAPS_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_YEAR_RE = re.compile(r"\b\d{4}\b")
_PCT_RE = re.compile(r"\b\d+(?:\.\d+)?%\b")

# Phrases too generic to be treated as grounding-critical (country names, etc.).
_GROUNDING_COMMON = {
    "New Zealand", "United States", "United Kingdom", "United Nations",
    "World War", "North America", "South America", "Middle East", "Far East",
}


def _extract_grounding_tokens(text: str) -> set:
    """Tokens whose provenance we care about when validating the answer."""
    out = set()
    for m in _CAPS_PHRASE_RE.finditer(text or ""):
        s = m.group(0)
        if s not in _GROUNDING_COMMON and len(s) > 2:
            out.add(s)
    for m in _YEAR_RE.finditer(text or ""):
        out.add(m.group(0))
    for m in _PCT_RE.finditer(text or ""):
        out.add(m.group(0))
    return out


def _evidence_corpus(messages) -> str:
    """Join external-evidence text: user messages (question + tool_responses) and system msg.
    Assistant <think> / <tool_call> are NOT evidence — they are model output and may themselves
    echo facts from removed rounds, which we specifically want to catch."""
    buf = []
    for m in messages:
        role = m.get("role")
        if role in ("user", "system"):
            c = m.get("content", "") or ""
            if isinstance(c, str):
                buf.append(c)
    return "\n".join(buf).lower()


def check_answer_grounding(new_messages, orig_messages):
    """Verify every grounding-critical token in the new <answer> (and the new final <think>)
    still traces back to external evidence (user messages: question + tool_responses).
    Returns (ok, reason_or_None)."""
    if not new_messages:
        return True, None
    final_content = new_messages[-1].get("content", "") or ""
    ans_match = ANSWER_RE.search(final_content)
    think_match = THINK_RE.search(final_content)
    payload = ""
    if ans_match:
        payload += ans_match.group(1) + "\n"
    if think_match:
        payload += think_match.group(1)
    tokens = _extract_grounding_tokens(payload)
    if not tokens:
        return True, None

    kept_evidence = _evidence_corpus(new_messages)
    orig_evidence = _evidence_corpus(orig_messages)

    lost = []
    for t in tokens:
        tl = t.lower()
        if tl not in kept_evidence and tl in orig_evidence:
            lost.append(t)
    if lost:
        return False, "grounding_loss:" + ",".join(sorted(lost)[:5])
    return True, None


def extract_question(messages):
    """Recover the user's question regardless of whether messages[0] is system or user."""
    if messages[0].get("role") == "system":
        return messages[1].get("content", "")
    text = messages[0].get("content", "")
    if "Input Question:" in text and "Input image:" in text:
        try:
            return text.split("Input Question:")[1].split("Input image:")[0].strip()
        except Exception:
            return text
    return text


def split_rounds(messages):
    """Return (question_text, middle_rounds, final_msg).

    middle_rounds: list of dicts {round_id, asst_idx, user_idx, asst_content, user_content}
                   with round_id = 2..N-1, asst_idx/user_idx are indices into messages.
    final_msg:     {"round_id": N, "asst_idx": len(messages)-1, "asst_content": ...}
    """
    # Locate the first message: conventionally messages[0] is user with the question,
    # but some records start with a system message — we just treat the first non-system
    # message as Round 1 and pair the rest.
    start = 0
    if messages and messages[0].get("role") == "system":
        start = 1

    # Rounds 2..N-1 pair (asst, user) starting from start+1.
    middle = []
    rid = 2
    i = start + 1
    while i + 1 < len(messages):
        a = messages[i]
        u = messages[i + 1]
        if a.get("role") != "assistant" or u.get("role") != "user":
            break
        middle.append({
            "round_id": rid,
            "asst_idx": i,
            "user_idx": i + 1,
            "asst_content": a.get("content", ""),
            "user_content": u.get("content", ""),
        })
        rid += 1
        i += 2

    if i >= len(messages) or messages[-1].get("role") != "assistant":
        return None  # malformed
    final = {
        "round_id": rid,
        "asst_idx": len(messages) - 1,
        "asst_content": messages[-1].get("content", ""),
    }
    return extract_question(messages), middle, final


def trim(text, head=900, tail=200):
    if text is None:
        return ""
    if len(text) <= head + tail + 40:
        return text
    return text[:head] + f"\n...[{len(text)-head-tail} chars elided]...\n" + text[-tail:]


def build_traj_string(q, middle, final):
    parts = [f"**Round 1** (user question):\n{trim(q, 1500, 200)}\n"]
    for r in middle:
        parts.append(
            f"\n**Round {r['round_id']}**:\nAssistant:\n{trim(r['asst_content'], 1200, 200)}\n\n"
            f"User (tool_response):\n{trim(r['user_content'], 900, 200)}\n"
        )
    parts.append(
        f"\n**Round {final['round_id']}** (final answer):\n{trim(final['asst_content'], 1500, 300)}\n"
    )
    return "\n".join(parts)


def patch_think(content, new_think_text):
    """Replace the content between the first <think>...</think> with new_think_text."""
    if not THINK_RE.search(content):
        # No <think> tag: prepend one
        return f"<think>{new_think_text}</think>\n" + content
    return THINK_RE.sub(
        lambda m: f"<think>{new_think_text}</think>", content, count=1
    )


def _count_image_tokens(content):
    if isinstance(content, str):
        return content.count("<image>")
    if isinstance(content, list):
        n = 0
        for part in content:
            if isinstance(part, dict):
                t = part.get("text") or ""
                if isinstance(t, str):
                    n += t.count("<image>")
        return n
    return 0


def filter_images_by_drop(messages, images, drop_indices):
    """Keep images whose originating msg idx is NOT in drop_indices.
    `images` is a flat list in the order of <image> tokens across `messages`."""
    out, cursor = [], 0
    for i, m in enumerate(messages):
        n = _count_image_tokens(m.get("content", "") if isinstance(m, dict) else "")
        if i not in drop_indices:
            out.extend(images[cursor:cursor + n])
        cursor += n
    return out


def apply_plan(messages, plan, middle, final):
    """Apply a {status, remove, patch_think} plan to produce new messages + meta info."""
    remove_set = set(int(r) for r in plan.get("remove", []))
    patches = {int(k): v for k, v in (plan.get("patch_think") or {}).items()}

    # Removals are only valid on middle rounds; final round is not removable.
    removable = {r["round_id"] for r in middle}
    remove_set &= removable

    # Patches are allowed on any kept round: middle rounds or the final round.
    # (Fix for previously dropped patch_think on the final round — ~3.8% of reduced samples.)
    patch_round_to_asst_idx = {r["round_id"]: r["asst_idx"] for r in middle}
    patch_round_to_asst_idx[final["round_id"]] = final["asst_idx"]

    # Skip deletion if the result would leave 0 middle rounds AND the answer isn't trivially
    # derivable; we let the LLM decide that via status, but fold in a minimal safety net.
    if len(remove_set) >= len(middle) and len(middle) > 0:
        return None, {"skipped": "remove_all_middle"}

    applied_patches = {}
    # Apply patches first (on the ORIGINAL message list, before removing anything).
    new_messages = list(messages)
    for rid, new_think in patches.items():
        if rid in remove_set:
            continue  # skip patches aimed at rounds we're about to delete
        ai = patch_round_to_asst_idx.get(rid)
        if ai is None:
            continue  # rid is out of range
        orig_content = new_messages[ai].get("content", "")
        new_messages[ai] = {
            **new_messages[ai],
            "content": patch_think(orig_content, str(new_think).strip()),
        }
        applied_patches[rid] = True

    # Now drop removed rounds (by asst_idx and user_idx).
    drop_indices = set()
    for r in middle:
        if r["round_id"] in remove_set:
            drop_indices.add(r["asst_idx"])
            drop_indices.add(r["user_idx"])
    new_messages = [m for idx, m in enumerate(new_messages) if idx not in drop_indices]

    # Validate: must still alternate properly and end with assistant.
    if not new_messages or new_messages[-1].get("role") != "assistant":
        return None, {"skipped": "final_not_assistant"}

    return new_messages, {
        "removed": sorted(remove_set),
        "patched_rounds": sorted(applied_patches.keys()),
        "drop_indices": sorted(drop_indices),
    }


def process_one(line_id, line_text, max_retries=4):
    try:
        d = json.loads(line_text)
    except Exception as e:
        return line_id, None, {"error": f"json_decode:{e}"}

    messages = d.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return line_id, None, {"error": "no_messages"}

    split = split_rounds(messages)
    if split is None:
        return line_id, None, {"error": "bad_structure"}
    q, middle, final = split

    if len(middle) == 0:
        # Nothing to prune; keep unchanged.
        d_out = dict(d)
        d_out["original_messages"] = messages
        d_out["cp_meta"] = {
            "status": "non_reducible", "removed": [], "patched": [],
            "reason": "no_middle_rounds", "raw_plan": None,
        }
        return line_id, d_out, None

    traj = build_traj_string(q, middle, final)
    prompt = USER_PROMPT_TEMPLATE.format(INPUT_TRAJ_STRING=traj)

    completion = call_openrouter(prompt, image_url=None, max_retries=max_retries)
    if "error" in completion:
        return line_id, None, {"error": f"llm:{completion['error'][:120]}"}

    raw = completion["choices"][0]["message"]["content"]
    try:
        plan = loads_llm_json(raw)
    except Exception as e:
        return line_id, None, {"error": f"parse_json:{e}", "raw": raw[:400]}

    status = plan.get("status", "")
    if status not in ("reduce", "non_reducible", "illogical"):
        return line_id, None, {"error": f"bad_status:{status}", "raw": raw[:400]}

    # For non_reducible / illogical, keep original messages unchanged but record status.
    if status != "reduce":
        d_out = dict(d)
        d_out["original_messages"] = messages
        d_out["cp_meta"] = {
            "status": status,
            "removed": [],
            "patched": [],
            "reason": plan.get("reason", ""),
            "raw_plan": plan,
        }
        return line_id, d_out, None

    new_msgs, applied = apply_plan(messages, plan, middle, final)
    if new_msgs is None:
        # Fall back to original
        d_out = dict(d)
        d_out["original_messages"] = messages
        d_out["cp_meta"] = {
            "status": "skipped",
            "removed": [],
            "patched": [],
            "reason": applied.get("skipped", ""),
            "raw_plan": plan,
        }
        return line_id, d_out, None

    # Grounding check: every proper-noun / year / percentage in the new <answer> must still
    # appear somewhere in the kept context (or the final <think>). If not, the LLM's removal
    # stripped away the only source of a fact the answer still states — roll back.
    ok, reason = check_answer_grounding(new_msgs, messages)
    if not ok:
        d_out = dict(d)
        d_out["original_messages"] = messages
        d_out["cp_meta"] = {
            "status": "non_reducible",
            "removed": [],
            "patched": [],
            "reason": reason,
            "raw_plan": plan,
        }
        return line_id, d_out, None

    d_out = dict(d)
    d_out["original_messages"] = messages
    orig_images = d.get("images")
    if isinstance(orig_images, list) and orig_images:
        d_out["images"] = filter_images_by_drop(
            messages, orig_images, set(applied.get("drop_indices", []))
        )
    d_out["messages"] = new_msgs
    d_out["cp_meta"] = {
        "status": "reduce",
        "removed": applied["removed"],
        "patched": applied["patched_rounds"],
        "reason": plan.get("reason", ""),
        "raw_plan": plan,
    }
    return line_id, d_out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/toolcall_3more_ocr_code_6200_cz_cleaned_1216.jsonl")
    ap.add_argument("--output_dir", default="./data_optim_cc")
    ap.add_argument("--sample_size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mp", type=int, default=20)
    ap.add_argument("--output_suffix", default="1500")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.output_suffix}" if args.output_suffix else ""
    out_file = os.path.join(args.output_dir, f"critical_path{tag}_" + os.path.basename(args.data))
    err_file = os.path.join(args.output_dir, f"critical_path{tag}.errors.jsonl")

    # Resume support
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done.add(int(json.loads(line).get("_idx", -1)))
                except Exception:
                    pass
    if os.path.exists(err_file):
        with open(err_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done.add(int(json.loads(line).get("_idx", -1)))
                except Exception:
                    pass
    done.discard(-1)
    print(f"已完成 {len(done)} 条")

    with open(args.data) as f:
        all_lines = list(enumerate(f))

    if args.sample_size and args.sample_size > 0:
        rng = random.Random(args.seed)
        picked = set(rng.sample(range(len(all_lines)), k=min(args.sample_size, len(all_lines))))
        pending = [(i, ln) for i, ln in all_lines if i in picked and i not in done]
        print(f"随机抽样 {len(picked)} 条 (seed={args.seed})，其中待处理 {len(pending)} 条")
    else:
        pending = [(i, ln) for i, ln in all_lines if i not in done]
        print(f"待处理 {len(pending)} 条")

    out_lock = threading.Lock()
    err_lock = threading.Lock()
    out_f = open(out_file, "a", buffering=1)
    err_f = open(err_file, "a", buffering=1)

    try:
        with ThreadPoolExecutor(max_workers=args.mp) as ex:
            futures = {ex.submit(process_one, lid, text): lid for lid, text in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), ncols=70):
                lid = futures[fut]
                try:
                    _, kept, err = fut.result()
                except Exception as e:
                    with err_lock:
                        err_f.write(json.dumps({"_idx": lid, "error": f"worker_exc:{type(e).__name__}:{str(e)[:120]}"}, ensure_ascii=False) + "\n")
                    continue
                if kept is not None:
                    kept["_idx"] = lid
                    with out_lock:
                        out_f.write(json.dumps(kept, ensure_ascii=False) + "\n")
                else:
                    rec = {"_idx": lid, **(err or {})}
                    with err_lock:
                        err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        out_f.close()
        err_f.close()

    clean_file = out_file.replace(".jsonl", ".clean_for_training.jsonl")
    n = clean_for_training(out_file, clean_file)
    sz = os.path.getsize(clean_file)
    print(f"{out_file} -> {clean_file}  ({n} lines, {sz/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
