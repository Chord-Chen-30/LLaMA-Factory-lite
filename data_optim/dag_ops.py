import collections
from collections import defaultdict


def get_dag_edges(edges):
    """检测并去掉成环的边，返回 DAG 边列表
    edges: list of dicts with keys 'source', 'target', 'reason'
    """

    # 邻接表
    adj = collections.defaultdict(list)
    for e in edges:
        adj[e['source']].append(e['target'])
    
    dag_edges = [] # list of (source, target) tuples
    visited, stack = set(), set()

    def dfs(u):
        visited.add(u)
        stack.add(u)
        # 排序保证稳定性
        for v in sorted(adj.get(u, [])):
            if v in stack:
                continue  # 跳过导致环的边
            dag_edges.append((u, v))
            if v not in visited:
                dfs(v)
        stack.remove(u)
    
    # 对所有节点进行 DFS，确保覆盖所有连通分量
    all_nodes = sorted({n for e in edges for n in (e['source'], e['target'])})
    for n in all_nodes:
        if n not in visited:
            dfs(n)
    return dag_edges

def prune_leaves(dag_edges):
    """
    持续移除出度为 0 的节点（保留 ID 最大的节点）
    dag_edges: list of (source, target) tuples
    返回: (pruned_edges, leaf_edges)
    """
    current_edges = list(dag_edges)

    while True:
        try:
            nodes = {int(n) for e in current_edges for n in e}
        except ValueError:
            nodes = {n for e in current_edges for n in e}

        if not nodes: break
        
        max_id_node = str(max(nodes))
        # print(nodes)
        # print(max_id_node)
        sources = {e[0] for e in current_edges}
        targets = {e[1] for e in current_edges}
        
        # 出度为 0 的节点定义：出现在 target 中但从未出现在 source 中
        leaves = {n for n in targets if n not in sources and n != max_id_node}
        
        if not leaves: 
            break

        # 过滤掉指向这些叶子节点的边
        # leaf_edges.extend([e for e in current_edges if e[1] in leaves])
        current_edges = [e for e in current_edges if e[1] not in leaves]
    
    leaf_edges = [e for e in dag_edges if e not in current_edges]
    return current_edges, leaf_edges

def merge_nodes_relaxed(edges):
    """
    Level 3: 合并有相同父节点集合、彼此无依赖关系的节点（不要求同子节点）。
    在 DAG 中 "同父节点集合" 已经隐含 "彼此无祖孙路径"（证明见讨论）。
    迭代执行直到收敛。
    edges: list of (source, target) tuples
    返回: list of (source, target) tuples with merged nodes.
    """
    parents = defaultdict(set)
    children = defaultdict(set)
    nodes = set()
    for u, v in edges:
        parents[v].add(u)
        children[u].add(v)
        nodes.update([u, v])

    while True:
        groups = defaultdict(list)
        for n in nodes:
            groups[frozenset(parents[n])].append(n)

        target = None
        for members in groups.values():
            if len(members) >= 2:
                target = members
                break
        if target is None:
            break

        new_label = "+".join(sorted(target, key=lambda x: (len(x), x)))
        new_parents = set()
        new_children = set()
        for m in target:
            new_parents |= parents.pop(m, set())
            new_children |= children.pop(m, set())
            nodes.discard(m)
        tset = set(target)
        new_parents -= tset
        new_children -= tset
        nodes.add(new_label)
        parents[new_label] = new_parents
        children[new_label] = new_children

        for other in list(nodes):
            if other == new_label:
                continue
            ps = parents.get(other, set())
            if ps & tset:
                parents[other] = (ps - tset) | {new_label}
            cs = children.get(other, set())
            if cs & tset:
                children[other] = (cs - tset) | {new_label}

    new_edges = set()
    for u, cs in children.items():
        for v in cs:
            new_edges.add((u, v))
    return sorted(new_edges)


def merge_nodes(edges):
    """
    合并有相同父节点、相同子节点、不相互依赖的节点
    edges: list of (source, target) tuples
    返回: list of (source, target) tuples with merged nodes. 比如： [("1", "2+3"), ("2+3", "4")]
    """

    # 1. 建立邻接关系
    parents = defaultdict(set)
    children = defaultdict(set)
    nodes = set()
    for u, v in edges:
        children[u].add(v)
        parents[v].add(u)
        nodes.update([u, v])

    # 2. 按 (父节点集, 子节点集) 分组
    groups = defaultdict(list)
    for n in nodes:
        # 使用 frozenset 是因为 set 不可哈希，不能做字典的 key
        key = (frozenset(parents[n]), frozenset(children[n]))
        groups[key].append(n)

    # 3. 生成节点映射表 {原节点: 新标签}
    mapping = {}
    for members in groups.values():
        # 如果组内有多个节点，用 '+' 连接，否则保持原样（或转字符串）
        label = "+".join(map(str, sorted(members))) if len(members) > 1 else members[0]
        for n in members:
            mapping[n] = label

    # 4. 重新构建边并去重
    new_edges = {(mapping[u], mapping[v]) for u, v in edges if mapping[u] != mapping[v]}
    
    return sorted(list(new_edges))

def process_pipeline(data):
    # 1. 去环
    dag = get_dag_edges(data)
    # 2. 修剪叶子
    pruned, leaf_edges = prune_leaves(dag)
    # 3. 合并节点
    merged = merge_nodes(pruned)
    return merged, pruned, leaf_edges


# --- 测试 ---
if __name__ == "__main__":
    raw_data = [
        {"source": "A", "target": "B"},
        {"source": "A", "target": "C"},
        {"source": "B", "target": "D"},
        {"source": "C", "target": "D"},
        {"source": "D", "target": "A"}, # 环
        {"source": "D", "target": "E"}, # 正常叶子
        {"source": "D", "target": "Z"}, # ID 最大的节点，应保留
        # {"source": "X", "target": "Y"}, # 孤立分支的叶子，应移除
    ]

    # raw_data = [{'source': '1', 'target': '2'}, {'source': '2', 'target': '3'}, {'source': '3', 'target': '4'}, {'source': '4', 'target': '5'}, {'source': '5', 'target': '6'}, {'source': '6', 'target': '7'}, {'source': '7', 'target': '8'}, {'source': '8', 'target': '9'}, {'source': '5', 'target': '7'}, {'source': '5', 'target': '8'}, {'source': '5', 'target': '9'}, {'source': '1', 'target': '5'},{'source': '1', 'target': '11'}, {'source': '9', 'target': '10'}, {'source': '10', 'target': '11'}]
    raw_data = [{'source': '1', 'target': '2'}, {'source': '2', 'target': '3'}, {'source': '1', 'target': '4'}, {'source': '3', 'target': '5'}, {'source': '4', 'target': '5'},]

    merged, pruned, leaf_edges = process_pipeline(raw_data)
    for edge in merged:
        print(edge)

    print("After pruning:")
    pruned.sort()
    for edge in pruned:
        print(edge)

    print('leaf_edges:')
    leaf_edges.sort()
    for edge in leaf_edges:
        print(edge)
