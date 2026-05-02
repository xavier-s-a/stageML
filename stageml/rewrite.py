from __future__ import annotations

import copy
import operator
from dataclasses import dataclass
from typing import Callable

import torch.fx as fx

from stageml.annotations import BindingTime, stage0, stage1
from stageml.tracer import propagate_stages


_MATMUL_TARGETS: set[Callable] = {
    operator.matmul,
}

try:
    import torch
    _MATMUL_TARGETS.add(torch.matmul)
    _MATMUL_TARGETS.add(torch.mm)
except Exception:
    torch = None


@dataclass(frozen=True)
class RewriteStats:
    lora_assoc_rewrites: int = 0
    total_rewrites: int = 0


def _is_matmul_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _MATMUL_TARGETS


def _stage_of(annotations: dict[fx.Node, BindingTime], node: fx.Node) -> BindingTime:
    return annotations.get(node, stage1)


def rewrite_static_right_assoc_matmul(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime], RewriteStats]:
    """
    Rewrite right-associated LoRA-style matmuls.

    Pattern:
        (x @ A_static) @ B_static

    Rewritten as:
        x @ (A_static @ B_static)

    Why this matters:
        Microsoft's loralib.Linear computes the LoRA branch as
        x @ A.T @ B.T. The first matmul touches x, so the whole chain becomes
        dynamic under normal staging analysis. This pass uses associativity to
        expose A.T @ B.T as a stage-0 computation, which StageML can then fold.

    Safety condition:
        The pass only fires when the left input of the inner matmul is stage-1
        and both right-side operands are stage-0. It does not change arithmetic
        order for chains where either static operand is dynamic.
    """
    original_gamma = _placeholder_gamma_from_annotations(gm, annotations)
    gm = copy.deepcopy(gm)
    annotations = propagate_stages(gm.graph, original_gamma)
    graph = gm.graph
    rewrites = 0

    for outer in list(graph.nodes):
        if not _is_matmul_node(outer):
            continue
        if len(outer.args) < 2:
            continue

        inner, rhs2 = outer.args[0], outer.args[1]
        if not isinstance(inner, fx.Node) or not isinstance(rhs2, fx.Node):
            continue
        if not _is_matmul_node(inner):
            continue
        if len(inner.args) < 2:
            continue

        x, rhs1 = inner.args[0], inner.args[1]
        if not isinstance(x, fx.Node) or not isinstance(rhs1, fx.Node):
            continue

        if _stage_of(annotations, x) != stage1:
            continue
        if _stage_of(annotations, rhs1) != stage0:
            continue
        if _stage_of(annotations, rhs2) != stage0:
            continue

        with graph.inserting_before(outer):
            combined = graph.call_function(operator.matmul, args=(rhs1, rhs2))
            combined.name = f"stage0_assoc_{rhs1.name}_{rhs2.name}"
            replacement = graph.call_function(operator.matmul, args=(x, combined))
            replacement.name = f"assoc_{outer.name}"

        outer.replace_all_uses_with(replacement)
        rewrites += 1

    if rewrites:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    new_annotations = propagate_stages(gm.graph, _placeholder_gamma_from_annotations(gm, annotations))
    stats = RewriteStats(lora_assoc_rewrites=rewrites, total_rewrites=rewrites)
    return gm, new_annotations, stats


def _placeholder_gamma_from_annotations(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> dict[str, BindingTime]:
    gamma: dict[str, BindingTime] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            gamma[node.name] = annotations.get(node, stage1)
    return gamma


def optimize_evaluation_order(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime], RewriteStats]:
    return rewrite_static_right_assoc_matmul(gm, annotations)
