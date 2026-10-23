"""Pure dependency planning before a continuation is admitted or dispatched.

Paths identify AST occurrences, including unnamed containers. They are local to
the frozen spec, not persisted execution coordinates or a dispatch permission.
"""
from __future__ import annotations

import posixpath
import re
from collections import defaultdict, deque
from dataclasses import dataclass

from .spec import AgentNode, BranchNode, ForeachNode, LoopNode, ParallelNode, SequenceNode

_TEMPLATE = re.compile(r"\{\{\s*([a-zA-Z_][\w-]*\.[a-zA-Z_]\w*)\s*\}\}")
_FILE = re.compile(r"@@FILEREF:([^\[\]@]+)(\[[^\]]*\])?@@")


def children(node):
    if isinstance(node, SequenceNode):
        return node.steps
    if isinstance(node, ParallelNode):
        return node.branches
    if isinstance(node, (LoopNode, ForeachNode)):
        return (node.body,)
    if isinstance(node, BranchNode):
        return (*node.cases.values(), *((node.default,) if node.default is not None else ()))
    if isinstance(node, AgentNode) and node.fallback is not None:
        return (node.fallback,)
    return ()


def walk(node, path=()):
    yield path, node
    for index, child in enumerate(children(node)):
        yield from walk(child, (*path, index))


def subtree_ids(node):
    return frozenset(n.id for _, n in walk(node) if n.id)


def within(path, ancestor):
    return path[:len(ancestor)] == ancestor


@dataclass(frozen=True)
class ContinuationPlan:
    target: str
    target_path: tuple[int, ...]
    affected_paths: frozenset[tuple[int, ...]]
    invalidated: frozenset[str]
    reasons: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self):
        return {"target": self.target, "target_path": list(self.target_path),
                "invalidated": sorted(self.invalidated),
                "affected_paths": [list(p) for p in sorted(self.affected_paths)],
                "reasons": {key: list(value) for key, value in self.reasons}}


def plan_continuation(spec, node_id):
    """Compute a monotone closure; never read sessions, run tools or mutate results.

    Sequence tails are conservatively invalidated, even without explicit refs.
    Parallel siblings remain reusable unless a declared/template/file dependency
    reaches them. Unknown external file/tool dependencies cannot be inferred.
    """
    from .engine import WorkflowRunError

    nodes = dict(walk(spec.root))
    by_id = {}
    for path, node in nodes.items():
        if node.id:
            if node.id in by_id:
                raise WorkflowRunError(f"ambiguous continuation node id {node.id}")
            by_id[node.id] = path
    target = by_id.get(node_id)
    if target is None or not isinstance(nodes[target], AgentNode):
        raise WorkflowRunError(f"unknown continuation agent {node_id}")
    for length in range(len(target)):
        ancestor = nodes[target[:length]]
        if isinstance(ancestor, ForeachNode):
            raise WorkflowRunError("foreach continuation needs an iteration-specific target")
        if isinstance(ancestor, AgentNode):
            raise WorkflowRunError("fallback continuation needs its owning agent target")

    # An edge marks a node or its entire subtree. Container changes caused by a
    # descendant do not invalidate independent siblings, unlike changed inputs.
    edges = defaultdict(list)
    files = defaultdict(list)
    for path, node in nodes.items():
        if isinstance(node, AgentNode) and node.output_file:
            files[posixpath.normpath(node.output_file)].append(path)
        for index, _child in enumerate(children(node)):
            child_path = (*path, index)
            edges[child_path].append((path, isinstance(node, ForeachNode), "aggregate"))
            if isinstance(node, SequenceNode):
                if index + 1 < len(node.steps):
                    edges[child_path].append(((*path, index + 1), True, "sequence"))
            elif isinstance(node, AgentNode):
                edges[path].append((child_path, True, "fallback"))

    def reference(ref, consumer, reason, *, deep=True):
        producer = by_id.get(ref.split(".", 1)[0])
        if producer is not None:
            edges[producer].append((consumer, deep, reason))

    for path, node in nodes.items():
        if isinstance(node, AgentNode):
            # inputs_from reads exact result keys; unlike ref fields it does not
            # resolve a dotted payload expression.
            for ref in node.inputs_from:
                if ref in by_id:
                    edges[by_id[ref]].append((path, True, "inputs_from"))
            if node.when:
                reference(str(node.when.get("ref") or ""), path, "when")
            for ref in _TEMPLATE.findall(node.input):
                reference(ref, path, "template")
            for name, _ in _FILE.findall(node.input):
                for producer in files.get(posixpath.normpath(name), ()):
                    edges[producer].append((path, True, "file"))
        elif isinstance(node, BranchNode):
            reference(node.on, path, "branch")
        elif isinstance(node, ForeachNode) and node.items_from:
            reference(node.items_from, path, "items_from")
        elif isinstance(node, LoopNode):
            ref = str(node.until.get("ref") or "")
            producer = by_id.get(ref.split(".", 1)[0])
            # An internal changed verdict recomputes the loop aggregate. An
            # external changed condition invalidates the whole loop execution.
            reference(ref, path, "until", deep=producer is None or not within(producer, path))

    affected, expanded = set(), set()
    reasons = defaultdict(set)
    queue = deque([(target, False, "target")])
    while queue:
        path, deep, reason = queue.popleft()
        reasons[path].add(reason)
        if deep and path not in expanded:
            expanded.add(path)
            for index, _child in enumerate(children(nodes[path])):
                queue.append(((*path, index), True, reason))
        if path in affected:
            continue
        affected.add(path)
        queue.extend(edges[path])

    invalid = frozenset(nodes[p].id for p in affected if nodes[p].id)
    return ContinuationPlan(node_id, target, frozenset(affected), invalid,
        tuple(sorted((nodes[p].id, tuple(sorted(why))) for p, why in reasons.items() if nodes[p].id)))


def replay_results(spec, plan, results):
    """Retain unaffected terminal evidence, including failures, without rerunning it.

    Targeted repair must not silently retry an unrelated failed side effect.
    Preserved failures still prevent the workflow from reporting success.
    """
    scope_ids = subtree_ids(spec.root)
    replay = {key: value for key, value in results.items()
              if key in scope_ids and key not in plan.invalidated and value.status not in ("pending", "running")}
    for _, node in walk(spec.root):
        if isinstance(node, ForeachNode) or (isinstance(node, LoopNode) and node.id in replay):
            for key in subtree_ids(node.body):
                replay.pop(key, None)
    return replay
