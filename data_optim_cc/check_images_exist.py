import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

F = Path("./data_optim_cc/critical_path_toolcall_3more_ocr_code_6200_cz_cleaned_1216.jsonl")
WORKERS = 64

paths = []
with F.open() as f:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        imgs = obj.get("images") or []
        if not isinstance(imgs, list):
            imgs = [imgs]
        for p in imgs:
            paths.append((lineno, obj.get("_idx"), p))

print(f"total paths: {len(paths)}  unique: {len(set(p[2] for p in paths))}")

def check(item):
    return item if not os.path.isfile(item[2]) else None

missing = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i, res in enumerate(ex.map(check, paths, chunksize=64), 1):
        if res is not None:
            missing.append(res)
        if i % 5000 == 0:
            print(f"  checked {i}/{len(paths)}  missing so far: {len(missing)}")

print(f"checked {len(paths)}  missing: {len(missing)} ({len(set(m[2] for m in missing))} unique)")
for item in missing[:20]:
    print("  line", item[0], "_idx", item[1], item[2])
if len(missing) > 20:
    print(f"  ... and {len(missing) - 20} more")

# flag any relative path too
rel = [p for p in paths if not p[2].startswith("/")]
if rel:
    print(f"!! {len(rel)} relative paths found (examples):")
    for p in rel[:5]:
        print("  ", p)
