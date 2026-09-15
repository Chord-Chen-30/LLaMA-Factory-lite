# Task 1 Report — Strict-Prompt DAG Labeling + Level 1/2/3 Pruning

## Summary

We replaced the original DAG-annotation prompt (too permissive — it marked chronological/narrative references as dependencies) with a stricter version that asks the LLM to identify only *globally load-bearing* dependencies toward the final answer. We also added a new pruning level (level 3, "relaxed merge": merges sibling rounds sharing the same parent set even when their children differ).

The strict prompt produces a much sparser DAG, which unlocks meaningful reductions in every pruning level — especially level 3.

## Per-level comparison

| Config | N | Avg msgs before | Avg msgs after | Msg reduction | Total chars reduction | Samples affected | Median msgs saved (affected) |
|---|---|---|---|---|---|---|---|
| loose-prompt level1 | 6194 | 10.24 | 10.20 | 0.33% | 0.14% | 85 (1.4%) | 2 |
| loose-prompt level2 | 6194 | 10.24 | 10.20 | 0.33% | 0.14% | 85 (1.4%) | 2 |
| strict-prompt level1 | 6196 | 10.23 | 9.77 | 4.50% | 3.64% | 986 (15.9%) | 2 |
| strict-prompt level2 | 6196 | 10.23 | 9.69 | 5.35% | 3.67% | 1175 (19.0%) | 2 |
| strict-prompt level3 | 6196 | 10.23 | 9.61 | 6.11% | 3.69% | 1322 (21.3%) | 2 |

> **Note:** Level 2/3 numbers above are from the **2026-04-27 regeneration** with the merge-format fix. The fix (a) skips merge groups that include round 1 — the user-question turn has no paired tool roundtrip and merging it produced OOD `<tool_response>` content; (b) rewrites the merge to always emit a single canonical `<think>...</think><tool_call>...</tool_call>` (or `<answer>`) on the assistant side and a single `<tool_response>...</tool_response>` on the user side. The cost is ~40% lower merge rate (level 3: 12.81% → 6.11%) in exchange for 100% format-clean training data. See [report_optim_level3_inference_blowup.md](report_optim_level3_inference_blowup.md) for the diagnosis that motivated this fix.

## Takeaways

1. **The prompt rewrite is the main unlock.** Under the loose (original) prompt, even level 2 only touches 1.4% of samples and reduces 0.33% of messages. Under the strict prompt, level 2 reduces ~5.4% of total messages and touches 19.0% of samples.

2. **Level 3 (relaxed merge) gives an additional boost over level 2** because the strict prompt creates many sibling rounds that share a common parent but feed different downstream consumers — exactly the case level 3 captures.

3. **Leaf-pruning (level 1) alone also benefits.** The strict prompt more aggressively marks dead-end tool calls as non-dependencies of the final answer, so the 'out-degree=0' set grows.

4. **Message count vs total chars — the gap is expected.** Levels 2 and 3 *merge* rounds (their content is concatenated into one round); they do not delete tokens. So 6.1% round reduction with only 3.7% char reduction is by design. For inference efficiency, round count is what matters (each round = one model forward pass + one tool roundtrip), so the 6.1% is the figure to quote for the user's stated goal of 'reducing unnecessary tool calls / thinking during inference'.

5. **Round-1 protection.** The post-fix merge logic refuses to merge any group that includes round 1 (the user's original question). In the previous run this was the source of all 1,461 multi-`<tool_response>` artifacts in level 3 training data — the strict-prompt LLM occasionally placed round 1 in the same root group as round 2, and the naïve merge then wrapped the question content inside `<tool_response>` tags, leaking system-prompt-shaped text into a position the agent loop would never produce at inference time.

## Strict-prompt Level 3 — distribution of messages saved per sample

| Msgs saved | # samples |
|---|---|
| 0 | 4874 |
| 2 | 832 |
| 4 | 392 |
| 6 | 79 |
| 8 | 12 |
| 10 | 7 |

## Qualitative example

- Sample `_idx=23`: 8 → 6 messages.

- Dependency summary (strict prompt + level3 merged):

  - `1 → 4`: Round 1 defines the core task of finding when the pictured mountain was renamed Dufourspitze, which Round 4 directly investigates.
  - `4 → 5`: Round 4 produced the load-bearing fact that the mountain was renamed on 28 January 1863, which Round 5 uses to conclude it was in the 19th century.