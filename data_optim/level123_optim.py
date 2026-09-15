"""
得到 DAG 标注之后，
1. 读取大模型标注，删掉错误格式的边。
2. 建立映射 source-target -> reason
3. 对每条数据：
    a. 通过 DAG 操作（prune_leaves）得到需要删除的消息
    b. 删除消息（evict_leaf_nodes）
    c. level 2 / 3: merge_nodes / merge_nodes_relaxed 合并兄弟节点
    d. level 2a / 3a: 同 2 / 3，但合并时用 LLM 重写 <think> 内容（顺滑而非分号拼接）
    e. 生成新的边（包含合并后的节点）和新的消息列表
修改 messages 字段、添加 optim_level{level}_dependency 字段。
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dag_ops import get_dag_edges, prune_leaves, merge_nodes, merge_nodes_relaxed
from msg_ops import (
    evict_leaf_nodes,
    merge_nodes as merge_msg_nodes,
    merge_nodes_with_think_rephrasing,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent_infer" / "src"))
from utils import loads_llm_json

try:
    from rich import print
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_IMG_TOK = "<image>"


def _n_img_tokens(m):
    c = m.get("content", "") if isinstance(m, dict) else ""
    return c.count(_IMG_TOK) if isinstance(c, str) else 0


def _build_images_per_msg(messages, images):
    out, cursor = [], 0
    for m in messages:
        n = _n_img_tokens(m)
        out.append(list(images[cursor:cursor + n]))
        cursor += n
    if cursor != len(images):
        raise ValueError(f"<image> token count {cursor} != len(images) {len(images)}")
    return out


def _apply_merge_to_images(images_per_msg, merged_edges):
    """Mirror msg_ops.merge_nodes on the parallel images_per_msg list."""
    merged_labels = set()
    for src, tgt in merged_edges:
        if "+" in src:
            merged_labels.add(src)
        if "+" in tgt:
            merged_labels.add(tgt)
    for label in merged_labels:
        rids = sorted(int(x) for x in label.split("+"))
        first = rids[0]
        fa, fu = first * 2 - 3, first * 2 - 2
        for rid in rids[1:]:
            a, u = rid * 2 - 3, rid * 2 - 2
            images_per_msg[fa].extend(images_per_msg[a])
            images_per_msg[a] = []
            images_per_msg[fu].extend(images_per_msg[u])
            images_per_msg[u] = []


parser = argparse.ArgumentParser()
parser.add_argument("--level", type=str, default="1", choices=["1", "2", "3", "2a", "3a"],
                    help="1=leaf-prune; 2=strict merge; 3=relaxed merge; "
                         "2a/3a=same as 2/3 but LLM-rephrase merged <think>.")
parser.add_argument("--data", type=str, required=True,
                    help="JSONL with a `dependency_raw` field from one_time_prompting.py.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for optim_level{level}_*.jsonl. Defaults to dirname(--data).")
parser.add_argument("--mp", type=int, default=1,
                    help="Concurrency for sample processing. Use 1 for level 1/2/3 (no LLM); "
                         "16+ for level 2a/3a so LLM rephrase calls run in parallel.")

args = parser.parse_args()

# Lazy LLM rephraser — only loaded when level 2a / 3a is requested, so users who
# only want 1/2/3 don't need OPENROUTER_API_KEY etc. configured.
_USE_REPHRASE = args.level in ("2a", "3a")
if _USE_REPHRASE:
    from merge_think_llm import rephrase_thinks  # noqa: E402

output_dir = args.output_dir or os.path.dirname(args.data)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    f"optim_level{args.level}_"+os.path.basename(args.data)
)

tmp_analysis_file = os.path.join(
    output_dir,
    "tmp",
    f"optim_level{args.level}_"+os.path.basename(args.data)
)
os.makedirs(os.path.dirname(tmp_analysis_file), exist_ok=True)


with open(args.data) as f:
    data = [json.loads(line) for line in f]


def _remove_round(str_):
    return str_.replace("Round ", "").replace("tool_call", "").replace("_", "").strip().split(' ')[0]

# Read edges. Invalid-format samples are kept with their original messages untouched.
valid_data = []
invalid_data = []
err_log = defaultdict(int)
for d in data:
    try:
        deps = loads_llm_json(d["dependency_raw"])
        refined_deps = []
        for dep in deps:
            dep["source"] = _remove_round(dep["source"])
            dep["target"] = _remove_round(dep["target"])
            refined_deps.append(dep)
        d["dependency_raw"] = refined_deps
        valid_data.append(d)
    except Exception as e:
        err_log[str(e)] += 1
        invalid_data.append(d)

logger.info(f"Valid data: {len(valid_data)}  Passthrough (unparseable DAG): {len(invalid_data)}")
print(f"Error log: ")
for k, v in err_log.items():
    print(f"  - {v}: {k}")
# print(valid_data[0]['dependency_raw'])


# Build source-target -> reason mapping
source_target_to_reason = []
for d in valid_data:
    source_target_to_reason_map = {}
    for dep in d["dependency_raw"]:
        source_target_to_reason_map[(dep["source"], dep["target"])] = dep["reason"]
    source_target_to_reason.append(source_target_to_reason_map)


def _process_sample(idx, d, source_target_to_reason_map):
    """Process one valid sample. Returns (idx, d_modified, item_pruned, item_merged,
    prev_len, cur_len). Pure per-sample logic — safe to run in a thread pool."""
    d['original_messages'] = copy.deepcopy(d['messages'])
    prev_len = len(d['original_messages'])

    _orig_images = list(d.get('images') or [])
    images_per_msg = _build_images_per_msg(d['messages'], _orig_images)

    dag_edges = get_dag_edges(d["dependency_raw"])  # Remove cycles

    # Prune leaves
    pruned_edges, leaf_edges = prune_leaves(dag_edges)
    item_pruned = len(dag_edges) - len(pruned_edges)
    new_messages = evict_leaf_nodes(d['messages'], leaf_edges)

    for _src, _tgt in leaf_edges:
        _tr = int(_tgt)
        images_per_msg[2 * _tr - 3] = []
        images_per_msg[2 * _tr - 2] = []

    # Pick merge variant by level
    if args.level == "2":
        merged = merge_nodes(pruned_edges)
        new_messages = merge_msg_nodes(new_messages, merged)
        _apply_merge_to_images(images_per_msg, merged)
    elif args.level == "3":
        merged = merge_nodes_relaxed(pruned_edges)
        new_messages = merge_msg_nodes(new_messages, merged)
        _apply_merge_to_images(images_per_msg, merged)
    elif args.level == "2a":
        merged = merge_nodes(pruned_edges)
        new_messages = merge_nodes_with_think_rephrasing(new_messages, merged, rephrase_thinks)
        _apply_merge_to_images(images_per_msg, merged)
    elif args.level == "3a":
        merged = merge_nodes_relaxed(pruned_edges)
        new_messages = merge_nodes_with_think_rephrasing(new_messages, merged, rephrase_thinks)
        _apply_merge_to_images(images_per_msg, merged)
    else:  # level "1"
        merged = pruned_edges

    item_merged = len(pruned_edges) - len(merged)

    # Generate new edges with reasons (output-side; same logic for all levels)
    json_edges = []
    for edge in merged:
        source = edge[0]
        target = edge[1]
        reasons = []
        for s in source.split('+'):
            for t in target.split('+'):
                _reason = source_target_to_reason_map.get((s, t))
                if _reason and _reason not in reasons:
                    reasons.append(_reason)
        json_edges.append({"source": source, "target": target, "reason": ';\n'.join(reasons)})

    d[f'optim_level{args.level}_dependency'] = json_edges
    d['images'] = [p for i, m in enumerate(new_messages) if m is not None for p in images_per_msg[i]]
    d['messages'] = [m for m in new_messages if m is not None]
    cur_len = len(d['messages'])
    return idx, d, item_pruned, item_merged, prev_len, cur_len


g = open(output_file, 'w')
g_ana = open(tmp_analysis_file, 'w')

num_leaves_pruned = 0
num_merged = 0
previous_msg_len = 0
current_msg_len = 0

assert len(source_target_to_reason) == len(valid_data)

# Run per-sample processing in a thread pool (LLM calls in level 2a/3a are I/O bound).
# Results are buffered and written in input order so the output file matches the input.
results_cache = {}
next_to_write = 0

def _flush_in_order():
    global next_to_write, num_leaves_pruned, num_merged, previous_msg_len, current_msg_len
    while next_to_write in results_cache:
        _idx, _d, _ip, _im, _pl, _cl = results_cache.pop(next_to_write)
        num_leaves_pruned += _ip
        num_merged += _im
        previous_msg_len += _pl
        current_msg_len += _cl
        line = json.dumps(_d, ensure_ascii=False) + '\n'
        g.write(line)
        if _ip > 0 or _im > 0:
            g_ana.write(line)
        next_to_write += 1

with ThreadPoolExecutor(max_workers=max(1, args.mp)) as ex:
    futures = {
        ex.submit(_process_sample, i, d, m): i
        for i, (d, m) in enumerate(zip(valid_data, source_target_to_reason))
    }
    done_count = 0
    total_to_do = len(futures)
    for fut in as_completed(futures):
        result = fut.result()
        idx = result[0]
        results_cache[idx] = result
        _flush_in_order()
        done_count += 1
        if done_count % 100 == 0 or done_count == total_to_do:
            logger.info(f"processed {done_count}/{total_to_do} samples")

# Drain anything left (shouldn't happen, but be safe)
_flush_in_order()

# Pass-through: samples whose DAG couldn't be parsed are emitted unchanged so the
# output line count matches the input. They get original_messages for parity and
# an empty optim_level{level}_dependency field.
for d in invalid_data:
    d['original_messages'] = copy.deepcopy(d['messages'])
    d[f'optim_level{args.level}_dependency'] = []
    previous_msg_len += len(d['original_messages'])
    current_msg_len += len(d['messages'])
    g.write(json.dumps(d, ensure_ascii=False) + '\n')

g.close()
g_ana.close()

total = len(valid_data) + len(invalid_data)
print(f"Leaves pruned: {num_leaves_pruned}")
print(f"Merged: {num_merged}")
print(f"Passthrough (unparseable DAG, original messages kept): {len(invalid_data)}")
print(f"Previous message length: {previous_msg_len/total}")
print(f"Current message length: {current_msg_len/total}")


# Sidecar: strip debug/meta fields so HF `datasets` can infer a stable schema.
KEEP_FOR_TRAINING = {"messages", "images", "system"}
clean_file = output_file.replace(".jsonl", ".clean_for_training.jsonl")
n_clean = 0
with open(output_file) as _fin, open(clean_file, "w") as _fout:
    for _line in _fin:
        if not _line.strip():
            continue
        _d = json.loads(_line)
        _d2 = {k: v for k, v in _d.items() if k in KEEP_FOR_TRAINING}
        _fout.write(json.dumps(_d2, ensure_ascii=False) + "\n")
        n_clean += 1
print(f"Clean-for-training written: {clean_file} ({n_clean} lines)")