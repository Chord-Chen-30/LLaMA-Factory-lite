# Task 2 Report — LLM-Based Critical-Path Pruning (Scheme B)

## Approach

For each trajectory we feed the full rounds (including the final answer) to `openai/gpt-5.4` and ask it to:
1. Tag each middle round KEEP/REMOVE by judging *globally* whether the round's output is actually used in the final answer.
2. Emit minimal `<think>` rewrites for any kept round whose opening would otherwise dangle (e.g. 'let's try another link') after an adjacent deletion — including the final answer round.
3. Return an overall status: `reduce | non_reducible | illogical`.

Every removal plan is gated by a **grounding check**: the new `<answer>` (plus the new final `<think>`) is scanned for proper-noun phrases, years, and percentages; each such token must still appear somewhere in the kept *user-role* evidence (question + tool_responses). If a token disappears because the source round was removed, the whole plan is rolled back and the sample is marked `non_reducible`. This guards against the common LLM failure mode of deleting a round whose facts are still echoed verbatim in the answer.

- Prompt: [prompt_cp.py](prompt_cp.py)
- Main script: [critical_path.py](critical_path.py)

## Summary

| Metric | Value |
|---|---|
| Samples processed (written to output) | 6138 (+ 58 worker errors) |
| `status=reduce` | 4447 (72.5%) |
| `status=non_reducible` | 1560 (25.4%) |
| — of which rolled back by **grounding check** | 797 |
| `status=illogical` | 8 (0.1%) |
| `status=skipped` (local safety net) | 123 (2.0%) |
| Avg msgs before | 10.24 |
| Avg msgs after | 7.55 |
| **Total msg reduction** | **26.32%** |
| Total char reduction | 22.28% |
| `<think>` char reduction | 28.86% |
| Median msgs saved among reduced | 2 |
| Max msgs saved in one sample | 24 |
| Samples receiving a continuity `patch_think` | 3179 / 4447 |
| — of which patched the **final round** (fix A) | 1626 |

## Distribution of messages saved per sample (reduced samples only)

| Msgs saved | # samples |
|---|---|
| 2 | 2418 |
| 4 | 1057 |
| 6 | 536 |
| 8 | 227 |
| 10 | 111 |
| 12 | 52 |
| 14 | 20 |
| 16 | 15 |
| 18 | 6 |
| 20 | 2 |
| 22 | 1 |
| 24 | 1 |

## Comparison with Task 1 (strict-prompt DAG + level 3)

Both methods operate on the full `toolcall_3more_ocr_code_6200_cz_cleaned_0421` set (~6196 samples). Task 1 (DAG-based, level 3 relaxed-merge) *merges* rounds: fewer rounds but content is concatenated, so total chars are largely preserved. Task 2 (critical-path) *deletes* rounds: fewer rounds AND fewer tokens, at the cost of one extra LLM call per sample (the labeling call).

| Method | N | Avg msgs before→after | Msg reduction | Char reduction | `<think>` reduction |
|---|---|---|---|---|---|
| Task 1 strict-prompt level 3 | 6196 | 10.23 → 8.92 | 12.81% | 4.00% | ≈0% |
| Task 2 critical-path (v2 with A+B fixes) | 6138 | 10.24 → 7.55 | 26.32% | 22.28% | 28.86% |

## Grounding rollback examples (samples LLM wanted to reduce, check blocked)

- `_idx=50`: would-remove=[4]; **rollback reason** `grounding_loss:Lunar Calendar,The People`
- `_idx=48`: would-remove=[3]; **rollback reason** `grounding_loss:Holy Name Cathedral`
- `_idx=112`: would-remove=[4]; **rollback reason** `grounding_loss:Domain Name System`

## Error breakdown

| Error class | Count |
|---|---|
| worker_exc | 48 |
| no_messages | 9 |
| parse_json | 1 |

## Final-round patch examples (previously silently dropped; fix A)

- `_idx=103` removed=[4] patched rounds=[5] (msgs saved 2)
- `_idx=141` removed=[3, 4] patched rounds=[5] (msgs saved 4)

## Large reductions (≥ 6 msgs saved)

- `_idx=2001` saved=6 removed=[2, 4, 6] patched=[3, 5]
  - reason: Round 2 图像检索无结果且后续未使用；Round 4 只是对已确定的蜂鸟分类做二次确认，信息冗余；Round 6 再次验证日历结论，前文已得到足够证据，可删。保留轮次经最小改写后连贯，不影响最终答案。
- `_idx=2030` saved=6 removed=[2, 3, 5] patched=[4, 6]
  - reason: 第2、3轮是失败/无关的图像与泛化搜索，未进入后续关键证据链；第5轮只是再次泛搜子科支持度，真正用于结论的是第6轮的具体论文结论。

## Takeaways

1. **Critical-path pruning remains the more aggressive lever.** Even after the grounding rollback filter takes ~15% of would-be reductions off the table (returning them as non_reducible), the net reduction is ~26% on rounds and ~22% on tokens, versus Task 1's 12.8% rounds / 4.0% tokens.

2. **The grounding check was worth adding.** It rolled back 797 / (4447 + 797) ≈ 15.2% of the samples the LLM had marked `reduce` because the surviving `<answer>` still named entities only present in removed rounds — exactly the 'unjustified fact' regression we found on spot-check. Affected samples stay as the untouched original, preserving SFT quality at the cost of a smaller reduction rate.

3. **Final-round patch fix has substantial measurable effect.** 1626 / 4447 ≈ 36.6% of reduced samples now carry a cleanup of the final `<think>`'s opening that previously would have been dropped.

4. **LLM residual errors are low (<1%).** Out of ~6196 inputs, only 58 fail with worker exceptions (mostly `'list' object has no attribute 'get'` from unusual LLM return structures); standalone retries succeed, so these are transient and safe to leave as drops.
