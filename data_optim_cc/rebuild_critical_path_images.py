"""One-time rebuild of `images` field for the critical_path non-clean output.

Step 1. Load msg_hash -> full-images map from the fixed 0421 source
        (/data/toolcall_3more_ocr_code_6200_cz_cleaned_0421.jsonl).

Step 2. For each row in critical_path non-clean output:
        - Overwrite images with the full list (abs paths) looked up by hash(original_messages).
        - Filter images based on cp_meta.removed (drop indices of removed rounds' msgs).

Step 3. Write back, verify images count == <image> token count per row.
"""
import json
import hashlib
import re
from pathlib import Path

SRC_0421 = Path("./data/toolcall_3more_ocr_code_6200_cz_cleaned_0421.jsonl")
TGT = Path("./data_optim_cc/critical_path_toolcall_3more_ocr_code_6200_cz_cleaned_0421.jsonl")
TMP = TGT.with_suffix(TGT.suffix + ".tmp")

TOKEN_RE = re.compile(r"<image>")

def msg_hash(msgs):
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def count_image_tokens(content):
    if isinstance(content, str):
        return len(TOKEN_RE.findall(content))
    if isinstance(content, list):
        return sum(
            len(TOKEN_RE.findall(p.get("text") or ""))
            for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return 0

def msg_indices_for_round(round_id):
    """Return (asst_idx, user_idx) for middle round r (>=2)."""
    return 2 * round_id - 3, 2 * round_id - 2

# ---- Step 1: build msg_hash -> images map ----
src_map = {}
with SRC_0421.open() as f:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        src_map[msg_hash(obj["messages"])] = obj.get("images")
print(f"loaded {len(src_map)} rows from 0421 source")

# ---- Step 2+3: rewrite critical_path ----
rows = 0
patched = 0
unmatched = []
mismatches = []
with TGT.open() as f_in, TMP.open("w") as f_out:
    for lineno, line in enumerate(f_in, 1):
        if not line.strip(): continue
        rows += 1
        obj = json.loads(line)

        orig = obj.get("original_messages")
        if not orig:
            f_out.write(line)
            continue

        h = msg_hash(orig)
        full_images = src_map.get(h)
        if full_images is None:
            unmatched.append((lineno, obj.get("_idx")))
            f_out.write(line)
            continue

        cp = obj.get("cp_meta") or {}
        removed_rounds = cp.get("removed") or []
        drop = set()
        for r in removed_rounds:
            a, u = msg_indices_for_round(int(r))
            drop.add(a); drop.add(u)

        # Filter full_images
        out, cursor = [], 0
        for i, m in enumerate(orig):
            n = count_image_tokens(m.get("content", "") if isinstance(m, dict) else "")
            if i not in drop:
                out.extend(full_images[cursor:cursor + n])
            cursor += n

        # Sanity: number of <image> tokens in pruned messages should equal len(out)
        pruned_tokens = sum(
            count_image_tokens(m.get("content", "")) for m in obj.get("messages") or []
        )
        if pruned_tokens != len(out):
            mismatches.append((lineno, obj.get("_idx"), pruned_tokens, len(out),
                               len(full_images), cursor, sorted(drop), removed_rounds))

        obj["images"] = out
        patched += 1
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"rows={rows}  patched={patched}  unmatched_hash={len(unmatched)}  mismatches={len(mismatches)}")
for x in unmatched[:10]:
    print("  unmatched:", x)
for x in mismatches[:10]:
    print("  mismatch(line,_idx,tok,img,full,cursor,drop,removed):", x)

if unmatched or mismatches:
    print("NOT replacing; leaving .tmp for inspection")
else:
    TMP.replace(TGT)
    print(f"wrote: {TGT}")
