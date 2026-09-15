import json
import hashlib
from pathlib import Path

SRC = Path("./data/toolcall_3more_ocr_code_6200_cz_cleaned_0421.jsonl")
TGT = Path("./data_optim_cc/critical_path_toolcall_3more_ocr_code_6200_cz_cleaned_1216.jsonl")
TMP = TGT.with_suffix(TGT.suffix + ".tmp")

def msg_hash(msgs):
    s = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

src_map = {}
dup_hashes = []
src_rows = 0
with SRC.open() as f:
    for line in f:
        if not line.strip():
            continue
        src_rows += 1
        obj = json.loads(line)
        h = msg_hash(obj["messages"])
        if h in src_map:
            dup_hashes.append(h)
        src_map[h] = obj.get("images")
print(f"source rows: {src_rows}  unique msg hashes: {len(src_map)}  duplicates: {len(dup_hashes)}")

tgt_rows = 0
matched = 0
unmatched = []
with TGT.open() as f_in, TMP.open("w") as f_out:
    for lineno, line in enumerate(f_in, 1):
        if not line.strip():
            continue
        tgt_rows += 1
        obj = json.loads(line)
        orig = obj.get("original_messages")
        if orig is None:
            unmatched.append((lineno, obj.get("_idx"), "no_original_messages"))
            f_out.write(line)
            continue
        h = msg_hash(orig)
        if h in src_map:
            obj["images"] = src_map[h]
            matched += 1
        else:
            unmatched.append((lineno, obj.get("_idx"), "no_hash_match"))
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"target rows: {tgt_rows}  matched: {matched}  unmatched: {len(unmatched)}")
for item in unmatched[:20]:
    print("  ", item)

if unmatched:
    print("ABORT: unmatched rows present; leaving .tmp for inspection")
else:
    TMP.replace(TGT)
    print(f"wrote in place: {TGT}")
