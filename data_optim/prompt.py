"""
DAG annotation prompts.

LOOSE_PROMPT: initial version, tends to over-label edges (chronological
              proximity and local reference both trigger edges).
STRICT_PROMPT: stricter version. Only mark an edge when the upstream round has
               a GLOBAL contribution toward deriving the final answer, not
               merely because the downstream round textually references or
               narratively follows it.
"""


LOOSE_PROMPT = """\
# Role
You are an expert in logical reasoning and causal analysis. Your task is to analyze a multi-round AI Agent trajectory and deconstruct it into a Directed Acyclic Graph (DAG) that represents the true dependencies between interaction rounds.

# Task Description
Given a trajectory organized by "Rounds," you must identify which previous rounds provided the necessary information, context, or logical basis for a subsequent round to occur.

# Node Definition
Each node in the graph corresponds to a **Round ID** (e.g., Round 1, Round 2, ... Round N).
- **Round 1** typically contains the initial User Query.
- **Round 2 to N-1** typically contain the Assistant's thoughts, tool calls, and the resulting tool responses.
- **Round N** typically contains the Assistant's final reasoning and the ultimate answer.

# Definition of "Dependency" (Edge: Round i -> Round j)
A dependency exists if and only if **Round j** requires information generated or retrieved in **Round i**.
1. **Data Dependency**: Round j uses specific facts, identifiers, or data points that were first introduced or returned in Round i.
2. **Logical Dependency**: The reasoning or strategy in Round j is a direct response to the outcome of Round i (e.g., correcting an error from Round i, or proceeding to the next step of a plan formulated in Round i).
3. **Strictness**: Do NOT create an edge simply because Round i precedes Round j chronologically. If Round 3 only needs the original user request from Round 1 and does not rely on anything from Round 2, the only edge should be `Round 1 -> Round 3`.

# Output Format
Provide the result as a JSON edge list. Each entry must include the source, the target, and a brief justification.

Format:
[
  {{"source": "Round X", "target": "Round Y", "reason": "Brief justification"}},
  ...
]

# Input Trajectory
{INPUT_TRAJ_STRING}

# Analysis and Output
"""


STRICT_PROMPT = """\
# Role
You are an expert in logical reasoning and causal analysis. Your task is to analyze a multi-round AI Agent trajectory and deconstruct it into a Directed Acyclic Graph (DAG) that captures **only the dependencies that actually matter for producing the final answer**.

# Task Description
The trajectory is organized by "Rounds." You will be told which round contains the final answer (typically the last round). Your job is to decide, for every candidate edge `Round i -> Round j`, whether Round j would have been *impossible or clearly wrong* without Round i's concrete contribution — judged **globally against the final answer**, not by local narrative flow.

# Node Definition
Each node corresponds to a **Round ID** (Round 1, Round 2, ..., Round N).
- **Round 1**: the initial User Query.
- **Round 2 to N-1**: assistant thought + tool call + tool response.
- **Round N**: the final assistant reasoning and answer.

# Strict Definition of a Dependency Edge (Round i -> Round j)
An edge exists **only if ALL of the following are true**:
1. **Concrete carry-over**: A specific fact, number, entity, URL, identifier, or inferred conclusion that Round j actually *uses* was first produced in Round i — and cannot be obtained from any other earlier round or from the original query alone.
2. **Globally load-bearing**: Removing Round i would break Round j's ability to make progress toward the final answer in Round N. If Round j could have been produced by skipping Round i (perhaps with minor rewording), do NOT add the edge.
3. **Not mere narrative / chronological continuity**: Do NOT add an edge just because Round j's `<think>` text mentions, reacts to, or rhetorically references Round i. Reactions like "the previous search was not helpful, let me try again" are NOT dependencies if Round j's actual tool call and outcome do not reuse any concrete output from Round i.

# Anti-Patterns — Do NOT Add an Edge When:
- **Dead-end rounds**: Round i returned information that was not used anywhere downstream (including Round N). In that case i has no outgoing edges at all.
- **Redundant confirmation**: Round j merely re-verifies a fact already established in an earlier round; the verified fact entered the answer via the earlier round, not via Round j.
- **Parallel independent lookups**: Round i and Round j are both sub-queries derived from Round 1 (or from a common ancestor), with no information flowing from one to the other. In that case both depend on the common ancestor, not on each other.
- **Pure stylistic/continuity reference**: Round j says "continuing from the previous step" without actually consuming any output of Round i.

# Be Aggressive About Pruning Edges
Default to NOT adding an edge. When in doubt, ask: "If I deleted Round i from the transcript, would Round j still reach the same conclusion (perhaps via trivial rewording)?" If yes — do not add the edge. The resulting DAG should be the **minimum skeleton** that still explains how information flows into the final answer in Round N.

# Output Format
Provide the result as a JSON edge list. Each entry must include the source, the target, and a brief justification that names the concrete fact/artifact carried from i to j.

Format:
[
  {{"source": "Round X", "target": "Round Y", "reason": "Round X produced <specific fact> which Round Y uses to <specific purpose>"}},
  ...
]

# Input Trajectory
{INPUT_TRAJ_STRING}

# Analysis and Output
"""

MERGE_THINK_PROMPT = """\
Your task is to merge multiple <think></think> segments from a trajectory of model outputs into a single coherent reasoning passage.
Context: These <think> segments originally came from separate turns in a multi-turn user-assistant conversation, where each <think> block was followed by tool_call arguments. It has now been determined that the think contents and the tool_calls can each be consolidated into a single turn. Your job is to merge the think segments into one logical, coherent reasoning passage.
Requirements:

Preserve the original intent and logical flow of the reasoning.
If the original think segments reference or describe tool_calls (tool invocations), you must retain that meaning — do not drop references to which tools are being used or why.
Smooth out the transitions: the current think contents are crudely concatenated with semicolons (;) and line breaks. Rewrite them so they read as one continuous, natural chain of thought rather than disjoint fragments.
Do not add new reasoning that wasn't present in the original; only restructure and connect what is already there.
Compress for token efficiency without losing information. Eliminate redundancy, repeated context, filler phrases, and verbose restatements across segments. Merge overlapping points — but every distinct piece of information, decision, and tool-call reference from the original must still be present in the output.

Input (concatenated think contents):
{CONCATED_THINK}

Output format: Plain text only. Do not wrap the output in any special tags (no <think>, no XML, no markdown code fences)."""