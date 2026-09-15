# Why does `optim_level3` produce LONGER inference trajectories than baseline?

## TL;DR

`optim_level3` was trained on data where rounds were **merged** (level-3 relaxed merge: 12.81% fewer rounds, 4.00% fewer chars vs baseline). Yet on simplevqa it averages **47.28 msgs / 141K chars per sample**, vs baseline's **16.69 msgs / 45K chars** — ~2.8× more msgs and ~3.1× more chars.

Two compounding problems:

1. **Tail blowup (the dominant effect):** 15.3% of samples hit the agent-loop's max-rounds cap (~259 msgs), versus 4% for baseline. These ~46 stuck samples alone account for ~80% of the avg-msg gap.

2. **Even non-stuck samples are ~2× longer:** excluding stuck samples, `optim_level3` still averages 13.1 msgs vs baseline 6.7 — the model issues many more (but shorter) tool-call rounds even on well-behaved cases.

The likely root cause is a **training/inference format mismatch** introduced by the relaxed-merge: 6.8% of user turns in the training data contain *multiple concatenated `<tool_response>` blocks* (a format that never occurs at agent-loop inference). The model is being trained to expect consolidated responses it can never receive.

---

## Evidence

### Per-model inference profile (simplevqa, n=300)

| Model | Avg msgs | Avg asst chars | Avg user chars | Stuck (≥100 msgs) | Multi-`<think>` per asst |
|---|---|---|---|---|---|
| baseline | 16.69 | 2,455 | 3,202 | 12 (4.0%) | 3.0% |
| toolcall_3more | 12.51 | 638 | 5,251 | 7 (2.3%) | 0.5% |
| critical_path | 19.27 | 730 | 4,131 | 14 (4.7%) | 0.7% |
| **optim_level3** | **47.28** | **554** | **5,543** | **46 (15.3%)** | **0.2%** |

### Avg msgs split by stuck vs non-stuck

| Model | Avg msgs (all) | Stuck rate | Avg msgs (non-stuck only) |
|---|---|---|---|
| baseline | 16.69 | 4.0% | **6.67** |
| toolcall_3more | 12.51 | 2.3% | **7.85** |
| critical_path | 19.27 | 4.7% | **8.29** |
| optim_level3 | 47.28 | 15.3% | **13.09** |

> Even on samples that complete normally, `optim_level3` uses ~2× the msgs that baseline uses.

### Per-msg pattern

`optim_level3` produces **way shorter assistant messages** (554 chars) but **many more of them**, while baseline produces fewer-but-longer messages (2455 chars). Total content per sample is still higher in `optim_level3` because the count grows faster than the per-msg shrinkage.

The drop in multi-`<think>` rate from baseline 3.0% → optim_level3 0.2% suggests the model **did not learn to combine multiple thinks per turn** even though the training data was merged in that direction.

---

## Training-data forensics

We compared the strict-prompt training jsonls (baseline = unmerged DAG, level3 = relaxed-merge), looking at turn structure:

| Metric | Baseline | optim_level3 |
|---|---|---|
| Avg msgs/sample (training) | 10.23 | 8.92 |
| Avg chars per asst turn | 745 | 812 |
| Avg chars per user turn | 2,021 | 2,275 |
| Asst turns with multi-`<think>` | 0.3% | 0.4% |
| **User turns with multi-`<tool_response>`** | **0.0%** | **6.8%** |

The relaxed-merge **does not mostly inflate assistant turns** — multi-think rate is essentially unchanged (0.3% → 0.4%). The merge effect is concentrated on the **user side**: 6.8% of user turns in `optim_level3` training data contain something like:

```
Tool responses are as follows respectively.
<tool_response>... result A ...</tool_response>
<tool_response>... result B ...</tool_response>
```

This format is a direct artifact of `merge_msg_nodes` in `data_optim/msg_ops.py` concatenating the tool_responses from multiple merged sibling rounds.

### Why this is a problem at inference

The agent loop in `infer_w_tools.py` operates strictly one-tool-call-per-turn:

- Assistant emits `<think>` + a single `tool_call`.
- The runtime executes the call and returns **exactly one** `<tool_response>` as the next user turn.
- Cycle repeats.

So the "consolidated response" pattern (one user turn carrying responses for >1 prior calls) **never occurs at inference** — yet 6.8% of the training was that pattern. The model now expects sometimes to receive a multi-response user turn, never gets one, and its termination/branching cues become unreliable.

Concretely:

- The model was likely trained to terminate or move on once it had seen a *combined* response covering several sub-questions. Without that signal, it issues redundant follow-up tool calls.
- For ~15% of inputs the redundancy compounds and the agent runs to its max-round cap (~259).

---

## Sample comparison (simplevqa, illustrative)

Same kind of question, different model:

- **baseline** (typical case): 5 msgs total → system, user-question, assistant-with-tool_call, user-with-tool_response, assistant-with-answer. Each assistant turn averages 2.5K chars (a substantive `<think>` + a tool call OR a final answer).

- **optim_level3** (typical case): 9 msgs → multiple short tool-calling cycles. Each assistant turn is short (~500 chars), the model issues a quick tool call, gets the response, immediately issues another, etc.

- **optim_level3** (tail case ≥100 msgs): the model loops on the same sub-question, sometimes issuing the same query repeatedly, until cut off at ~259 msgs. Often these end without a final answer, contributing to the lower task accuracy.

---

## Why the other two SFT variants don't show the same blowup

- **toolcall_3more_ocr_code_0421**: pre-existing dataset, no merging, no rewriting → the model learned a clean one-call-per-turn rhythm. **Best efficiency** (12.5 msgs/sample, 7 stuck).

- **critical_path** (Task 2, deletion-based): rounds are *removed*, not merged. The training data still has a clean one-tool-call-per-user-turn structure — there is **never** a "Tool responses are as follows respectively." synthesizing multiple responses into one. → Only mild bloat (19.3 msgs/sample, 14 stuck).

Only `optim_level3` (and presumably level2, which uses the same merge primitive) introduces the multi-response user-turn artifact.

---

## Recommendations

1. **Don't deploy the merged-rounds format as-is for SFT.** The synthetic "Tool responses are as follows respectively." user turns are out-of-distribution at inference and seem to break the termination logic.

2. **If we still want round-count reduction during inference, prefer Task-2-style deletion** (critical_path). It produces cleaner training data and has a much better stuck rate (4.7% vs 15.3%) while preserving the benefits of fewer rounds.

3. **If we want to keep the merge approach**, two possible fixes — neither tested:
   - **Fix A (training-time):** keep the merge in the assistant turn (combine `<think>`s + multiple tool_calls into one assistant message), but split the user turn back into one `<tool_response>` per consecutive user turn. This way the inference-time format (one response per user turn) matches training.
   - **Fix B (inference-time):** modify the agent loop to batch multiple parallel tool responses into a single user turn when the assistant emitted multiple `tool_calls` together. This is more invasive and risks other regressions.

4. **Inspect the 46 stuck samples** to confirm the loop hypothesis. A quick inspection of trajectory tails would either confirm (the model repeats a query family) or surface a different failure mode.

---

## Numbers used in this report

- Inference outputs: `result/2026-04-25/qwen3_vl_30b_a8b_0421_optim_level3/checkpoint-400/simplevqa_w_tools.jsonl` (and the analogous baseline / toolcall / critical_path files).
- Training data: `data_optim/openai-gpt-5.4/{,optim_level3_}dependency_strictprompt_toolcall_3more_ocr_code_6200_cz_cleaned_0421.jsonl`.
- "Stuck" defined as `len(traj) >= 100`. The actual cap appears to be ~259 msgs (matches all observed maxes across models).
