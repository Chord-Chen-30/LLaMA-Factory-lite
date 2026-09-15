# optim_level3 Inference Bloat — Post-Format-Fix Analysis (2026-04-28)

## TL;DR

After the 0427 format fix (no more multi-`<tool_response>` artifacts, no stray "Tool responses are as follows" intro), `optim_level3` still produces **2× more tokens** and **86% more messages** per sample than `optim_level2`, despite scoring within 0.001 of it on the 4-dataset average.

The cause is **not** longer-per-turn output — each assistant turn is actually slightly shorter than `optim_level2`'s. The cause is **more turns + tripled stuck rate**. The aggressive sibling-merging in level 3 disrupts the linear "think → call → response → think" causal flow that the model relies on for termination cues.

**Recommendation:** ship `optim_level2`, drop `optim_level3` from the rotation. If we want to keep the relaxed-merge idea, restrict it to size-2 merges of time-adjacent siblings.

---

## Cross-model leaderboard (4 datasets, 1101 samples total)

| Model | AVG score | AVG msg | AVG tok | tok/msg | tok per 1 score-point |
|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B-Thinking | 0.4468 | 41.67 | 8,977 | 216 | ~20,100 |
| Raw trajectory | **0.4835** | 23.94 | 20,940 | 875 | ~43,300 |
| critical_path | 0.4432 | 27.33 | 19,540 | 715 | ~44,100 |
| **optim_level2** | 0.4822 | **22.25** | **15,841** | 712 | **~32,800** |
| optim_level3 | 0.4807 | 30.19 | **31,779** | 1,053 | ~66,100 |

`optim_level3` is the only SFT variant that costs *more* tokens per score point than the raw trajectory baseline.

---

## Diagnostic 1 — training data signatures are nearly identical

| Indicator (training data) | level2 | level3 |
|---|---|---|
| samples | 6,196 | 6,196 |
| avg msgs/sample | 9.69 | 9.61 |
| avg assistant chars/turn | 759 | 764 |
| avg `<think>` body | 559 | 564 |
| avg `<tool_call>` body | 160 | 162 |
| avg user `<tool_response>` body | 1,329 | 1,340 |

The two SFT datasets have indistinguishable per-turn shape. The difference must be in *how often merges happen* and *which rounds get combined*.

## Diagnostic 2 — level 3 triggers ~3× more merge events

| Merge group size | level 2 occurrences | level 3 occurrences | ratio |
|---|---|---|---|
| size 2 | 709 | **2,491** | 3.5× |
| size 3 | 230 | 586 | 2.5× |
| size 4 | 83 | 122 | 1.5× |
| size ≥5 | 22 | 35 | 1.6× |

The relaxed criterion (parents-only, ignoring children) catches many more sibling pairs than strict same-parent-same-children.

## Diagnostic 3 — inference-time shape (simplevqa, n=300)

| Indicator (inference) | optim_level2 | optim_level3 | level3 vs level2 |
|---|---|---|---|
| msgs/sample (avg) | 14.0 | **26.1** | +86% |
| msgs/sample (median) | 7 | 9 | +29% |
| **stuck rate (≥100 msgs)** | 7 (2.3%) | **20 (6.7%)** | 2.9× |
| assistant chars/turn (avg) | 644 | **591** | −8% |
| `<think>` chars/turn | 474 | 425 | −10% |
| `<tool_call>` chars/turn | 141 | 164 | +16% |
| user chars/turn | 4,062 | **5,917** | +46% |
| asst turns w/ multi-`<think>` or `;\n` style | 0% | 0% | — |

**Key reads:**

- The model **did not internalize the merge style** — 0% of inference assistant turns produce multiple `<think>` blocks or use the `;\n` separator characteristic of merged training data. Whatever level 3 was trying to teach about "compressing multiple sub-tasks into one turn", the model rejected it.
- Per-turn output is *shorter* in level 3, not longer. The whole bloat is from **more turns**.
- Stuck rate triples (2.3% → 6.7%) — these long-tail samples dominate the average. Without them, level 3's median is only 9 msgs vs 7 for level 2 (a small difference).
- User turns 46% longer means the tool responses are bigger — the model issues broader/more queries that each return more content.

## Hypothesised mechanism

Original training data has a clean linear flow:

```
think_2 → tool_call_2 → tool_response_2 → think_3 → tool_call_3 → ...
```

Each `think` is conditioned on the immediately prior `tool_response`, and each `tool_call` directly follows from its `think`. The model uses this strict alternation as the termination signal: when the latest `tool_response` matches what the question needs, emit `<answer>`.

Level 3 relaxed merge collapses sibling rounds whose only common dependency is the parent set:

```
                  ┌──→ think_3 + call_3
parent (round 2) ─┤
                  └──→ think_4 + call_4
```

In the merged training example, both `(think_3 + call_3)` and `(think_4 + call_4)` are concatenated into one assistant turn following the same `tool_response_2`. The user turn that follows holds *one* response (the post-fix format ensures this), but the model trained on this merged form loses the strict 1-to-1 mapping between calls and immediately-prior responses.

At inference, the agent loop is strictly 1-call-per-turn. The model:

1. Has weaker termination cues (since training showed multiple call patterns per `tool_response` block)
2. Has been conditioned to consider many parallel sub-tasks as one round
3. Issues each of those sub-tasks separately, expanding the conversation
4. Sometimes loops on the same sub-task family because the "is this enough?" check is fuzzier

This compounds into more turns and, on a non-trivial tail, full-blown stuck loops.

## Why level 2 is fine

Level 2 also merges, but with the strict criterion (same parent set AND same child set). This restricts merges to nodes that are truly interchangeable from a graph-flow perspective — they have the same upstream and the same downstream. The resulting training examples preserve the linear cause-effect chain at the boundaries of the merge. Hence ~3× fewer merge events, and the merges that do happen don't disturb the alternation pattern the model relies on.

## Solutions

### A — Drop level 3, keep level 2 ✅ recommended
Current data already supports this. Level 2 score is within 0.001 of level 3's average, with half the tokens and 26% fewer messages. No code changes needed.

### B — Restrict level 3 to size-2 merges
Modify `merge_nodes_relaxed` in [data_optim/dag_ops.py:72](../../data_optim/dag_ops.py#L72) to skip merge groups of size ≥ 3:

```python
target = None
for members in groups.values():
    if len(members) == 2:   # was: >= 2
        target = members
        break
```

Eliminates ~30% of merge events, keeps the bulk of round reduction, drops the highest-risk multi-round consolidations.

### C — Only merge time-adjacent siblings
Same function, additional filter:

```python
ids = sorted(int(x) for x in target if x.isdigit())
if any(ids[i+1] - ids[i] != 1 for i in range(len(ids)-1)):
    continue   # skip non-adjacent merges
```

Preserves chronological flow. Combine with B for safer level 3.

### D — Inference-time circuit breaker
Add to `infer_w_tools.py` agent loop:

- Track a hash of `(tool_name, args)` per call. If the same hash appears 2 times consecutively → force `<answer>` emission with a "stop reasoning, give best guess" prompt.
- Hard cap at e.g. 30 messages without seeing `<answer>` → force terminate.

This is a band-aid for the existing model — useful if we don't want to retrain.

### E — Curriculum training
Start training on level 1 (leaf-prune only, no merge) for 1–2 epochs to lock in the strict alternation pattern. Then continue on level 3 for the remaining epochs. Risk: the second phase may still erode the alternation prior. Cost: 2 training runs.

## Recommendation

1. **Today:** ship `optim_level2`. It dominates `optim_level3` on every cost metric while matching its score.
2. **Next iteration:** if we want a cheaper level-3-style training, try B alone first (size-2 only) — small change, easy to validate.
3. **Skip:** D (inference patches) — fragile; E (curriculum) — high effort, uncertain payoff.

## Numbers from

- Inference: `result/2026-04-28/qwen3_vl_30b_a8b_0427_optim_level{2,3}/checkpoint-400/*.jsonl`
- Training data: `data_optim/openai-gpt-5.4/2026-04-27/optim_level{2,3}_*.{jsonl,clean_for_training.jsonl}`
- Scoring: gpt-5-nano (`*_gpt5nano_eval.txt`)
- Tokenization: `Qwen2Tokenizer` from `models/Qwen3-VL-30B-A3B-Thinking` via `apply_chat_template`
