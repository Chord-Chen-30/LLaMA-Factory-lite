import copy
import re

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL)


def evict_leaf_nodes(messages, leaf_edges, cur_edges=None):
    try:
        new_messages = copy.deepcopy(messages)
        for leaf_edge in leaf_edges:
            _, target_round_id = leaf_edge
            target_round_id = int(target_round_id)
            msg_id_assistant = target_round_id * 2 - 3
            msg_id_user = target_round_id * 2 - 2
            new_messages[msg_id_assistant] = None
            new_messages[msg_id_user] = None
        return new_messages
    except Exception as e:
        print(e)
        return messages


def _extract_blocks(pattern, text):
    """Return all bodies (stripped) of every matched <tag>...</tag> pair. Empty list if none."""
    return [m.strip() for m in pattern.findall(text)]


def _strip_nested(text, *patterns):
    """Remove any matches of the given regex patterns from text. Used to scrub
    nested sibling tags out of an extracted body (e.g., a <tool_call> nested
    inside a <think> block in the source data)."""
    for p in patterns:
        text = p.sub('', text)
    return text.strip()


def _parse_assistant_content(content):
    """Parse assistant content into (thinks, tool_calls, answers).

    Each is a list of stripped bodies for every well-paired tag found. Lone/unbalanced
    tags (e.g., '<tool_response>' mentioned in prose) are ignored. If a sibling tag
    is nested inside a <think> body in the source (we have observed real cases of
    <tool_call> inside <think>), it is captured at the top level by its own regex
    AND its substring is scrubbed out of the think body to avoid duplicate emission
    on rebuild.
    """
    thinks_raw = _extract_blocks(_THINK_RE, content)
    tool_calls = _extract_blocks(_TOOL_CALL_RE, content)
    answers = _extract_blocks(_ANSWER_RE, content)
    thinks = [_strip_nested(t, _TOOL_CALL_RE, _ANSWER_RE) for t in thinks_raw]
    return thinks, tool_calls, answers


def _parse_user_content(content):
    """Parse user content into a list of <tool_response> bodies.

    If no well-paired <tool_response> tag is found (e.g., the round-1 question turn
    or content with stray '<tool_response>' mentions in prose), return the whole
    content as a single body — the caller decides whether to wrap it.
    """
    blocks = _extract_blocks(_TOOL_RESPONSE_RE, content)
    return blocks if blocks else [content.strip()]


def _build_assistant_content(thinks, tool_calls, answers):
    """Compose canonical assistant content.

    Output is exactly one of:
        <think>{thinks joined with ';\\n'}</think><tool_call>{calls joined with '\\n'}</tool_call>
        <think>{thinks joined with ';\\n'}</think><answer>{answers joined with '\\n'}</answer>

    If any <answer> body exists, the answer form is used (and any tool_calls are
    discarded since the merged turn is terminal). Otherwise the tool_call form is used.
    """
    think_body = ';\n'.join(t for t in thinks if t)
    if answers:
        body = '\n'.join(a for a in answers if a)
        return f'<think>{think_body}</think><answer>{body}</answer>'
    body = '\n'.join(c for c in tool_calls if c)
    return f'<think>{think_body}</think><tool_call>{body}</tool_call>'


def _build_user_content(blocks):
    """Compose canonical user content: a single <tool_response>{joined}</tool_response>.

    When 2+ blocks are joined (i.e. multiple original tool_responses are being
    merged into one user turn), a short English disclaimer is prepended so the
    model knows the body holds several prior tool returns in order. Single-block
    content is emitted bare (matches original training format).
    """
    blocks = [b for b in blocks if b]
    if len(blocks) >= 2:
        body = 'Previous tool call results:\n' + '\n'.join(blocks)
    else:
        body = '\n'.join(blocks)
    return f'<tool_response>{body}</tool_response>'


def _merge_assistant_content(keep_content, remove_content):
    # kt: keep thinks; ktc: keep tool_calls; ka: keep answers (should be mutually exclusive with ktc)
    kt, ktc, ka = _parse_assistant_content(keep_content)
    rt, rtc, ra = _parse_assistant_content(remove_content)
    return _build_assistant_content(kt + rt, ktc + rtc, ka + ra)


def _merge_user_content(keep_content, remove_content):
    kr = _parse_user_content(keep_content)
    rr = _parse_user_content(remove_content)
    return _build_user_content(kr + rr)


def _append_content(messages, keep_idx, remove_idx):
    """Append remove_idx's content into keep_idx, then mark remove_idx as None.

    Tolerates out-of-range indices (e.g., the final answer round has no paired
    user turn, so its user_idx may sit past the end of the list).
    """
    if not (0 <= keep_idx < len(messages)) or not (0 <= remove_idx < len(messages)):
        return
    keep, remove = messages[keep_idx], messages[remove_idx]
    if keep is None or remove is None:
        return

    if isinstance(keep, dict) and isinstance(remove, dict):
        kc = keep.get('content') or ''
        rc = remove.get('content') or ''
        # Multimodal (list-typed) content is not supported by the merge — leave as-is.
        if not isinstance(kc, str) or not isinstance(rc, str):
            return

        if keep.get('role') == 'assistant':
            keep['content'] = _merge_assistant_content(kc, rc)
        else:
            keep['content'] = _merge_user_content(kc, rc)

        # Carry over any structured tool_calls list.
        if 'tool_calls' in remove:
            keep.setdefault('tool_calls', []).extend(remove['tool_calls'])

    messages[remove_idx] = None


def _build_assistant_content_rephrased(thinks, tool_calls, answers, think_rephraser):
    """Like _build_assistant_content but, when 2+ thinks are present, runs them
    through the supplied callable to produce a single coherent rephrased body.

    think_rephraser receives the ;\\n-joined raw concatenation (matches what
    MERGE_THINK_PROMPT expects) and must return the rephrased text. On exception
    we fall back to the naive ;\\n join — same shape, no LLM.
    """
    thinks = [t for t in thinks if t]
    if len(thinks) <= 1:
        think_body = thinks[0] if thinks else ''
    else:
        concat = ';\n'.join(thinks)
        try:
            think_body = think_rephraser(concat).strip()
        except Exception as e:
            import logging
            logging.warning(f"think_rephraser failed ({e}); falling back to ;\\n join")
            think_body = concat
    if answers:
        body = '\n'.join(a for a in answers if a)
        return f'<think>{think_body}</think><answer>{body}</answer>'
    body = '\n'.join(c for c in tool_calls if c)
    return f'<think>{think_body}</think><tool_call>{body}</tool_call>'


def merge_nodes_with_think_rephrasing(messages, merged_edges, think_rephraser):
    """Variant of merge_nodes that asks an LLM to rephrase merged <think> bodies
    coherently instead of joining with ';\\n'.

    The rephraser is called ONCE per merge group (not pairwise), on the
    ';\\n'-joined concatenation of all thinks across rounds in that group.

    Other behaviour matches merge_nodes exactly:
    - Round-1 merge groups are skipped.
    - User <tool_response> bodies are joined with '\\n'.
    - <answer> dominates over <tool_call> if both are present in the merge.
    - Out-of-range / None / list-typed (multimodal) content is left alone.
    """
    new_messages = copy.deepcopy(messages)

    merged_labels = set()
    for src, tgt in merged_edges:
        if '+' in src:
            merged_labels.add(src)
        if '+' in tgt:
            merged_labels.add(tgt)

    for label in merged_labels:
        round_ids = sorted(int(x) for x in label.split('+'))
        if round_ids[0] < 2:
            continue

        all_thinks, all_tool_calls, all_answers = [], [], []
        all_user_blocks = []
        all_tool_calls_struct = []
        first_asst_idx = round_ids[0] * 2 - 3
        first_user_idx = round_ids[0] * 2 - 2

        for rid in round_ids:
            asst_idx = rid * 2 - 3
            user_idx = rid * 2 - 2
            if 0 <= asst_idx < len(new_messages):
                m = new_messages[asst_idx]
                if isinstance(m, dict):
                    c = m.get('content') or ''
                    if isinstance(c, str):
                        t, tc, a = _parse_assistant_content(c)
                        all_thinks.extend(t)
                        all_tool_calls.extend(tc)
                        all_answers.extend(a)
                    if 'tool_calls' in m:
                        all_tool_calls_struct.extend(m['tool_calls'])
            if 0 <= user_idx < len(new_messages):
                m = new_messages[user_idx]
                if isinstance(m, dict):
                    c = m.get('content') or ''
                    if isinstance(c, str):
                        all_user_blocks.extend(_parse_user_content(c))

        if 0 <= first_asst_idx < len(new_messages) and isinstance(new_messages[first_asst_idx], dict):
            new_messages[first_asst_idx]['content'] = _build_assistant_content_rephrased(
                all_thinks, all_tool_calls, all_answers, think_rephraser
            )
            if all_tool_calls_struct:
                new_messages[first_asst_idx]['tool_calls'] = all_tool_calls_struct
        if 0 <= first_user_idx < len(new_messages) and isinstance(new_messages[first_user_idx], dict):
            new_messages[first_user_idx]['content'] = _build_user_content(all_user_blocks)

        for rid in round_ids[1:]:
            asst_idx = rid * 2 - 3
            user_idx = rid * 2 - 2
            if 0 <= asst_idx < len(new_messages):
                new_messages[asst_idx] = None
            if 0 <= user_idx < len(new_messages):
                new_messages[user_idx] = None

    return new_messages


def merge_nodes(messages, merged_edges):
    """Merge multi-round nodes' messages into the first round of each group.

    messages: list (already passed through evict_leaf_nodes)
    merged_edges: edges with merged labels like [("1", "2+3"), ("2+3", "4")]

    Round 1 is the user question turn (no paired assistant), so any merge group
    that includes round 1 is skipped — merging the question with a tool_response
    would produce out-of-distribution content.
    """
    new_messages = copy.deepcopy(messages)

    merged_labels = set()
    for src, tgt in merged_edges:
        if '+' in src:
            merged_labels.add(src)
        if '+' in tgt:
            merged_labels.add(tgt)

    for label in merged_labels:
        round_ids = sorted(int(x) for x in label.split('+'))
        if round_ids[0] < 2:
            # Round 1 has no assistant turn (idx 2*1-3 = -1) and is the user
            # question, not a tool round. Skip such merges to keep the format
            # canonical.
            continue
        first = round_ids[0]
        first_asst_idx = first * 2 - 3
        first_user_idx = first * 2 - 2

        for rid in round_ids[1:]:
            asst_idx = rid * 2 - 3
            user_idx = rid * 2 - 2
            _append_content(new_messages, first_asst_idx, asst_idx)
            _append_content(new_messages, first_user_idx, user_idx)

    return new_messages


# --- Unit tests ---
if __name__ == "__main__":
    # Layout (round_id starts at 1; round 1 = user question only):
    #   idx 0: round1 user (initial question)
    #   idx 1: round2 assistant   idx 2: round2 user
    #   idx 3: round3 assistant   idx 4: round3 user
    #   idx 5: round4 assistant   idx 6: round4 user
    #   idx 7: round5 assistant (final answer)
    base_messages = [
        {"role": "user", "content": "初始问题。注：<tool_response> 出现在 user 回合"},
        {"role": "assistant", "content": '<think>思考2</think><tool_call>{"name":"search"}</tool_call>'},
        {"role": "user", "content": "<tool_response>result_2</tool_response>"},
        {"role": "assistant", "content": '<think>思考3</think><tool_call>{"name":"fetch"}</tool_call>'},
        {"role": "user", "content": "<tool_response>result_3</tool_response>"},
        {"role": "assistant", "content": '<think>思考4</think><tool_call>{"name":"calc"}</tool_call>'},
        {"role": "user", "content": "<tool_response>result_4</tool_response>"},
        {"role": "assistant", "content": '<think>思考5</think><answer>ANSWER</answer>'},
    ]

    def _new():
        return copy.deepcopy(base_messages)

    def _assert_canonical_assistant(content):
        """An assistant content must be EXACTLY <think>...</think> followed by either
        <tool_call>...</tool_call> or <answer>...</answer>, no extras, no nesting."""
        assert content.count('<think>') == 1, f"think open count != 1: {content!r}"
        assert content.count('</think>') == 1, f"think close count != 1: {content!r}"
        has_tc = '<tool_call>' in content
        has_ans = '<answer>' in content
        assert has_tc != has_ans, f"must have exactly one of tool_call/answer: {content!r}"
        if has_tc:
            assert content.count('<tool_call>') == 1
            assert content.count('</tool_call>') == 1
            assert re.fullmatch(r'<think>.*?</think><tool_call>.*?</tool_call>', content, re.DOTALL), f"asst tool_call shape wrong: {content!r}"
        else:
            assert content.count('<answer>') == 1
            assert content.count('</answer>') == 1
            assert re.fullmatch(r'<think>.*?</think><answer>.*?</answer>', content, re.DOTALL), f"asst answer shape wrong: {content!r}"

    def _assert_canonical_user(content):
        """A user content must be EXACTLY <tool_response>...</tool_response>, no extras."""
        assert content.count('<tool_response>') == 1, f"tool_response open count != 1: {content!r}"
        assert content.count('</tool_response>') == 1, f"tool_response close count != 1: {content!r}"
        assert re.fullmatch(r'<tool_response>.*?</tool_response>', content, re.DOTALL), f"user shape wrong: {content!r}"

    # Test 1: evict_leaf_nodes — drop round 3 (idx 3, 4)
    m = _new()
    out = evict_leaf_nodes(m, [("2", "3")])
    assert out[0] is not None and out[0]["role"] == "user"
    assert out[1] is not None and out[2] is not None and "result_2" in out[2]["content"]
    assert out[3] is None and out[4] is None
    assert out[5] is not None and "思考4" in out[5]["content"]
    assert "result_4" in out[6]["content"]
    assert "ANSWER" in out[7]["content"]
    assert m[3] is not None and m[4] is not None  # original untouched
    print("Test 1 passed: evict_leaf_nodes (single round)")

    # Test 2: evict_leaf_nodes — drop multiple rounds
    m = _new()
    out = evict_leaf_nodes(m, [("2", "3"), ("3", "4")])
    assert out[3] is None and out[4] is None
    assert out[5] is None and out[6] is None
    assert out[1] is not None and out[7] is not None
    print("Test 2 passed: evict_leaf_nodes (multi rounds)")

    # Test 3: merge round 3 + round 4 — content goes to round 3 slot
    m = _new()
    out = merge_nodes(m, [("2", "3+4"), ("3+4", "5")])
    assert out[5] is None and out[6] is None
    asst, user = out[3]["content"], out[4]["content"]
    _assert_canonical_assistant(asst)
    _assert_canonical_user(user)
    assert "思考3" in asst and "思考4" in asst
    assert ";\n" in asst  # think joined with ";\n"
    assert "fetch" in asst and "calc" in asst
    assert "result_3" in user and "result_4" in user
    assert user.index("result_3") < user.index("result_4")
    # untouched neighbours
    assert out[1]["content"] == m[1]["content"]
    assert out[7]["content"] == m[7]["content"]
    print("Test 3 passed: merge two rounds (canonical format, no duplicate tags)")

    # Test 4: merge three rounds 2+3+4
    m = _new()
    out = merge_nodes(m, [("1", "2+3+4"), ("2+3+4", "5")])
    assert out[3] is None and out[4] is None
    assert out[5] is None and out[6] is None
    asst, user = out[1]["content"], out[2]["content"]
    _assert_canonical_assistant(asst)
    _assert_canonical_user(user)
    for tag in ("思考2", "思考3", "思考4"):
        assert tag in asst
    for tag in ("search", "fetch", "calc"):
        assert tag in asst
    for r in ("result_2", "result_3", "result_4"):
        assert r in user
    assert user.index("result_2") < user.index("result_3") < user.index("result_4")
    print("Test 4 passed: merge three rounds")

    # Test 5: no merge label → messages unchanged
    m = _new()
    out = merge_nodes(m, [("1", "2"), ("2", "3")])
    assert all(out[i] is not None for i in range(len(m)))
    assert out == m
    print("Test 5 passed: no-op merge")

    # Test 6: combined evict-then-merge flow
    m = _new()
    evicted = evict_leaf_nodes(m, [("3", "4")])
    out = merge_nodes(evicted, [("1", "2+3"), ("2+3", "5")])
    assert out[3] is None and out[4] is None  # round 3 merged into round 2 slots
    assert out[5] is None and out[6] is None  # round 4 evicted
    asst, user = out[1]["content"], out[2]["content"]
    _assert_canonical_assistant(asst)
    _assert_canonical_user(user)
    assert "思考2" in asst and "思考3" in asst
    assert "result_2" in user and "result_3" in user
    assert "ANSWER" in out[7]["content"]
    print("Test 6 passed: evict + merge combined")

    # Test 7: REGRESSION — round 1 in merge group (the bug from level3 data).
    # Previously this would wrap the question content inside <tool_response> and
    # leave a stray opening tag from the question's prose. Now we skip the merge.
    m = _new()
    out = merge_nodes(m, [("1+2", "3")])
    # Round 1 + round 2 merge must be skipped → messages unchanged.
    assert out[0]["content"] == m[0]["content"], "round-1 merge must be skipped"
    assert out[1]["content"] == m[1]["content"]
    assert out[2]["content"] == m[2]["content"]
    # Stray <tool_response> in question prose stays as-is — never gets wrapped.
    assert "<tool_response>" in out[0]["content"]
    print("Test 7 passed: round-1 merge is skipped (regression for level3 bug)")

    # Test 8: merge of an answer-bearing round with a tool_call round
    # (unusual topology, but possible) → must produce <answer> form, no <tool_call>.
    custom = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": '<think>T2</think><tool_call>CALL2</tool_call>'},
        {"role": "user", "content": "<tool_response>R2</tool_response>"},
        {"role": "assistant", "content": '<think>T3</think><answer>ANS3</answer>'},
    ]
    out = merge_nodes(custom, [("1", "2+3")])
    asst = out[1]["content"]
    _assert_canonical_assistant(asst)
    assert '<answer>' in asst and '<tool_call>' not in asst, f"answer must dominate: {asst!r}"
    assert "T2" in asst and "T3" in asst
    assert "ANS3" in asst
    print("Test 8 passed: answer dominates over tool_call when merged")

    # Test 9: idempotency / re-parsing already-merged content.
    # Build a 4-round merge by repeated application — second call sees already-merged content.
    m = _new()
    # round 2+3+4: should still produce one canonical pair on each side.
    out = merge_nodes(m, [("1", "2+3+4")])
    asst, user = out[1]["content"], out[2]["content"]
    _assert_canonical_assistant(asst)
    _assert_canonical_user(user)
    # Now feed the merged content through the parser once more to verify idempotency.
    parsed_thinks, parsed_calls, parsed_ans = _parse_assistant_content(asst)
    assert len(parsed_thinks) == 1 and len(parsed_calls) == 1 and parsed_ans == []
    parsed_responses = _parse_user_content(user)
    assert len(parsed_responses) == 1
    print("Test 9 passed: merged output is canonical and re-parses to single bodies")

    # Test 10: multimodal (list-typed) content — merge should leave it untouched.
    multi = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": '<think>T</think><tool_call>X</tool_call>'},
        {"role": "user", "content": [{"type": "text", "text": "<tool_response>multi</tool_response>"}]},
        {"role": "assistant", "content": '<think>U</think><tool_call>Y</tool_call>'},
        {"role": "user", "content": "<tool_response>R3</tool_response>"},
    ]
    out = merge_nodes(multi, [("1", "2+3")])
    # The user-side merge into idx 2 should be skipped (list content), so idx 4
    # is unchanged from its prior position. Assistant-side merge proceeds normally.
    assert out[3] is None  # round-3 assistant slot cleared
    assert isinstance(out[2]["content"], list)  # list-typed user untouched
    _assert_canonical_assistant(out[1]["content"])
    print("Test 10 passed: multimodal content gracefully skipped in user-side merge")

    # Test 11: REGRESSION — source content has <tool_call> NESTED inside <think>.
    # Real example from training data: think body wraps a tool_call. Our parser
    # must extract the tool_call separately and scrub it from the think body so
    # the rebuilt content has exactly one <tool_call> pair.
    nested = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content":
            '<think>thinking_2</think><tool_call>CALL_2</tool_call>'},
        {"role": "user", "content": "<tool_response>R2</tool_response>"},
        {"role": "assistant", "content":
            '<think>thinking_3 NESTED <tool_call>NESTED_CALL</tool_call> trailing</think><tool_call>CALL_3</tool_call>'},
        {"role": "user", "content": "<tool_response>R3</tool_response>"},
    ]
    out = merge_nodes(nested, [("1", "2+3")])
    asst = out[1]["content"]
    _assert_canonical_assistant(asst)
    # Both legitimate top-level tool_calls captured + the nested one too,
    # so all three end up in the merged tool_call body — but as ONE pair of tags.
    assert "thinking_2" in asst and "thinking_3" in asst
    assert "trailing" in asst  # surrounding think text preserved
    assert "<tool_call>" not in asst.split('<think>', 1)[1].split('</think>', 1)[0], \
        "tool_call must NOT remain nested inside <think> body"
    assert "CALL_2" in asst and "CALL_3" in asst and "NESTED_CALL" in asst
    print("Test 11 passed: nested <tool_call> inside <think> is scrubbed and re-emitted once")

    # ============================================================
    # Detailed verbose tests for merge_nodes_with_think_rephrasing
    # ============================================================
    def hr(label):
        print("\n" + "=" * 78)
        print(f" {label}")
        print("=" * 78)

    def show_msgs(msgs, prefix=""):
        for i, m in enumerate(msgs):
            if m is None:
                print(f"  {prefix}[{i}] None (evicted/merged)")
            else:
                c = m.get("content", "")
                role = m.get("role", "?")
                snippet = (c[:200] + "...") if isinstance(c, str) and len(c) > 200 else c
                print(f"  {prefix}[{i}] {role}: {snippet!r}")

    # Test 12 — 3-way merge: rephraser called ONCE per group
    hr("Test 12: 3-way merge group (2+3+4) — rephraser called once, not pairwise")
    call_log = []
    def fake_rephraser(concat):
        call_log.append(concat)
        return f"REPHRASED({concat.count(';')+1} thinks)"

    m = _new()
    print("\nINPUT messages (8 total: round1-user + 4 (asst,user) pairs + final asst):")
    show_msgs(m, prefix="IN  ")
    print("\nMerge edges: [('1', '2+3+4'), ('2+3+4', '5')]   <- rounds 2,3,4 merge into round 2's slot")
    print("\nEXPECTED:")
    print("  - rephraser called exactly 1 time (not 2 like pairwise)")
    print("  - rephraser input contains '思考2;\\n思考3;\\n思考4'")
    print("  - out[1].content has 'REPHRASED(3 thinks)' inside <think>")
    print("  - out[1] tool_call body joins 'search' + 'fetch' + 'calc'")
    print("  - out[2] tool_response body joins 'result_2' + 'result_3' + 'result_4'")
    print("  - out[3], out[4], out[5], out[6] all None (evicted slots)")

    out = merge_nodes_with_think_rephrasing(m, [("1", "2+3+4"), ("2+3+4", "5")], fake_rephraser)
    print(f"\nACTUAL rephraser call_log (len={len(call_log)}):")
    for i, c in enumerate(call_log):
        print(f"  call #{i+1}: {c!r}")
    print(f"\nACTUAL output messages:")
    show_msgs(out, prefix="OUT ")

    asst = out[1]["content"]; user = out[2]["content"]
    _assert_canonical_assistant(asst); _assert_canonical_user(user)
    assert out[3] is None and out[4] is None and out[5] is None and out[6] is None
    assert len(call_log) == 1, f"expected 1 rephraser call, got {len(call_log)}"
    assert "思考2" in call_log[0] and "思考3" in call_log[0] and "思考4" in call_log[0]
    assert call_log[0].count(";\n") == 2
    assert "REPHRASED(3 thinks)" in asst
    assert "search" in asst and "fetch" in asst and "calc" in asst
    assert "result_2" in user and "result_3" in user and "result_4" in user
    print("\n✓ Test 12 passed")

    # Test 13 — no merge groups (no '+' in labels) → no rephraser calls, output unchanged
    hr("Test 13: no merge groups → rephraser must NOT be invoked")
    call_log.clear()
    m = _new()
    print("\nINPUT messages: same fixture as test 12")
    print("Merge edges: [('1', '2'), ('2', '3')]   <- NO '+' anywhere, nothing to merge")
    print("\nEXPECTED:")
    print("  - rephraser called 0 times (no merge groups)")
    print("  - output identical to input (no slot becomes None)")

    out = merge_nodes_with_think_rephrasing(m, [("1", "2"), ("2", "3")], fake_rephraser)
    print(f"\nACTUAL rephraser call_log: {call_log} (len={len(call_log)})")
    print(f"ACTUAL output == input? {out == m}")
    assert len(call_log) == 0
    assert out == m
    print("\n✓ Test 13 passed")

    # Test 14 — rephraser raises → fallback to ';\n' join
    hr("Test 14: rephraser exception → graceful fallback to ';\\n' join")
    def broken_rephraser(_concat):
        raise RuntimeError("simulated LLM failure")

    m = _new()
    print("\nINPUT messages: same fixture as test 12")
    print("Merge edges: [('2', '3+4'), ('3+4', '5')]   <- 2-way merge of rounds 3,4 into slot 3")
    print("rephraser: ALWAYS raises 'simulated LLM failure'")
    print("\nEXPECTED:")
    print("  - WARNING logged about think_rephraser failure")
    print("  - out[3].content has '思考3;\\n思考4' (raw concat, not 'REPHRASED')")
    print("  - format still canonical: exactly 1 <think>...</think><tool_call>...</tool_call>")
    print("  - out[5], out[6] still get evicted to None\n")

    out = merge_nodes_with_think_rephrasing(m, [("2", "3+4"), ("3+4", "5")], broken_rephraser)
    print("ACTUAL output messages:")
    show_msgs(out, prefix="OUT ")
    asst = out[3]["content"]
    _assert_canonical_assistant(asst)
    assert "思考3" in asst and "思考4" in asst and ";\n" in asst
    assert "REPHRASED" not in asst   # rephraser threw, so its return value never reached the output
    assert out[5] is None and out[6] is None
    print("\n✓ Test 14 passed")

    # Test 15 — round-1 merge group skipped, rephraser NOT called
    hr("Test 15: merge group containing round 1 → skipped, no rephraser call")
    call_log.clear()
    m = _new()
    print("\nINPUT messages: same fixture as test 12")
    print("Merge edges: [('1+2', '3')]   <- proposes merging round 1 (the user question) with round 2")
    print("\nEXPECTED:")
    print("  - merge skipped because round 1 has no paired assistant turn")
    print("  - rephraser called 0 times")
    print("  - all messages unchanged")
    out = merge_nodes_with_think_rephrasing(m, [("1+2", "3")], fake_rephraser)
    print(f"\nACTUAL rephraser call_log: {call_log} (len={len(call_log)})")
    print(f"ACTUAL out[0].content unchanged? {out[0]['content'] == m[0]['content']}")
    print(f"ACTUAL out[1].content unchanged? {out[1]['content'] == m[1]['content']}")
    print(f"ACTUAL out[2].content unchanged? {out[2]['content'] == m[2]['content']}")
    assert len(call_log) == 0
    assert out[0]["content"] == m[0]["content"]
    print("\n✓ Test 15 passed")

    # Test 16 — answer dominates over tool_call when merging
    hr("Test 16: merge mid-round (tool_call) with final round (answer) → output uses <answer>")
    custom = [
        {"role": "user",      "content": "Q"},
        {"role": "assistant", "content": '<think>T2</think><tool_call>CALL2</tool_call>'},
        {"role": "user",      "content": "<tool_response>R2</tool_response>"},
        {"role": "assistant", "content": '<think>T3</think><answer>ANS3</answer>'},
    ]
    print("\nINPUT messages:")
    show_msgs(custom, prefix="IN  ")
    print("\nMerge edges: [('1', '2+3')]   <- merge round 2 (tool_call) with round 3 (answer)")
    print("\nEXPECTED:")
    print("  - rephraser called 1 time on 'T2;\\nT3'")
    print("  - merged asst uses <answer> form (answer wins over tool_call)")
    print("  - 'CALL2' (the original tool_call body) is DROPPED (answer is terminal)")
    print("  - out[1] has REPHRASED + ANS3; out[3] is None")
    call_log.clear()
    out = merge_nodes_with_think_rephrasing(custom, [("1", "2+3")], fake_rephraser)
    print(f"\nACTUAL rephraser call_log: {call_log}")
    print("ACTUAL output messages:")
    show_msgs(out, prefix="OUT ")
    asst = out[1]["content"]
    _assert_canonical_assistant(asst)
    assert '<answer>' in asst and '<tool_call>' not in asst
    assert "REPHRASED(2 thinks)" in asst and "ANS3" in asst
    assert "CALL2" not in asst
    assert len(call_log) == 1
    print("\n✓ Test 16 passed")

    print("\n" + "=" * 78)
    print(" All 16 tests passed ✓")
    print("=" * 78)
