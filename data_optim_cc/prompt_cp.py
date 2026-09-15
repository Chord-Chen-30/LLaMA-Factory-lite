"""
Critical-Path Pruning prompt (Scheme B).

Unlike the DAG-based approach, this prompt works directly on the trajectory +
final answer. The LLM is asked to mark each middle round as KEEP or REMOVE
based on whether the round actually contributed to deriving the final answer.
For each REMOVE decision, the LLM also produces a minimal rewrite of the NEXT
kept round's <think> so the transcript remains coherent after deletions (no
dangling references like "let me try another link").

The prompt is built in Chinese to match the existing project style
(filter_rounds.py) and to keep the LLM firmly anchored to the removal task.
"""


SYS_PROMPT = (
    "你是一个严谨、保守的多模态 Agent 研究数据清洗助手。"
    "你的任务是剪掉不必要的推理/工具调用轮次，使训练数据更高效，"
    "同时保证剩下的轨迹仍然逻辑连贯、上下文自洽。"
    "严格按要求输出紧凑 JSON，不要加任何 markdown 代码块标记。"
)


USER_PROMPT_TEMPLATE = """\
# 背景
下面是一条 Agent 解题轨迹。每个中间 round 由一条 assistant（含 <think> 和 <tool_call>）+ 一条 user（tool_response）组成。最后一条 assistant 给出 <answer>。

# 目标
判断哪些中间 round **可以整段删除**，使得在不改变最终答案正确性的前提下，轨迹更简洁、更高效。

# 可删除 round 的典型特征
1. **失败/空结果**：tool_response 是 Page not found、空结果、无关内容，且后续没有基于此调整策略的关键用法。
2. **冗余重复**：tool_call 与之前某轮重叠，得到的信息已有；或多次搜索同一事实。
3. **被放弃的探索分支**：该 round 尝试了某个方向，但后续 think/tool_call 没有使用其产出，最终答案也与之无关。
4. **多余的二次验证**：关键事实已在别处首次得到并进入最终答案，本轮只是再确认一遍。
5. **过度冗长的思考准备**：该 round 的 tool_call 输出实际没进入最终答案的证据链。

# 不可删除 round 的特征
- 其工具产出的具体事实（数字、年份、人名、链接、图像描述等）**直接出现在最终 answer 中**；
- 或后续某个保留 round 的 think/tool_call 参数里显式引用了本轮产出；
- 或它是推理链上不可跳过的一环（如"先识别图中植物 → 再查该科化石"中的第一步）。

# 连贯性硬约束
删除某个 round 后，紧随其后的第一个保留 round 的 <think> 开头**不能**读起来像指向已删除内容，例如："let's try another link"、"那条链接没用"、"那不对，再搜"、"既然…不行"、"another search"、"换一条"、"上一次搜索"、"hmm that didn't work" 等。

为解决这种 dangling 引用，你可以在输出中为受影响的保留 round 提供一个**最小改写**（`patch_think`）：改写后的 <think> 必须：
- 不改变它的结论或随后的 tool_call；
- 去掉对已删除 round 的任何明显回指（上述那些短语）；
- 尽量重用原文措辞，只改抬头一两句；
- 可留空（`null`）表示无需改写。

# 保守原则
- 宁可多留，不要误删。如果你不确定一个 round 是否被最终答案用到，**保留**它。
- 如果整个轨迹里没有任何 round 可以安全删除，返回 `remove: []`。

# 输出格式
严格输出以下 JSON（不要 markdown 代码块）：
{{
  "status": "reduce" | "non_reducible" | "illogical",
  "remove": [<round_id_int>, ...],
  "patch_think": {{
      "<round_id_int>": "<rewritten think content, plain text without <think> tags>",
      ...
  }},
  "reason": "<简短中文说明>"
}}

- `status=reduce`：至少删 1 个且保证连贯，`remove` 给出要删除的 round id 列表。
- `status=non_reducible`：所有 round 都关键，或删掉任何 round 都会破坏连贯性且无法通过 patch_think 修复。此时 `remove=[]`、`patch_think={{}}`。
- `status=illogical`：原轨迹本身不合理（答案不对题、final answer 缺失、思考严重混乱）；`remove=[]`、`patch_think={{}}`。

- `remove` 中不应包含 Round 1（用户问题）或最后一个 assistant（最终答案）；只针对中间的 round。
- `patch_think` 的 key 必须是一个将被保留的 round id（int），且其原 <think> 的抬头确实需要去掉对已删 round 的回指；否则不要写进来。

# 输入轨迹
{INPUT_TRAJ_STRING}

# 你的 JSON 输出
"""
