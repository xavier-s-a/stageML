"""
stageml/tracer.py
Phase 2 — Stage Propagation

Takes the staging environment Γ from Phase 1 and a torch.fx graph.
Propagates binding-time annotations through every node in the graph.

The propagation rule (the whole analysis in one line):
    stage(node) = S   if ALL operands of node are S
    stage(node) = D   if ANY operand of node is D

This is a forward dataflow analysis over the two-point lattice {S ⊑ D}.

Week 2 goal: get this running on TinyMLP and print the annotated graph.
"""

from __future__ import annotations
from typing import Callable, Any
import torch
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1


# ── Stage-annotated FX node ───────────────────────────────────────────────────

def propagate_stages(
    graph: fx.Graph,
    gamma: dict[str, BindingTime]
) -> dict[fx.Node, BindingTime]:
    """
    Forward dataflow analysis.
    
    Given:
      graph : the torch.fx computation graph of the function
      gamma : staging environment Γ (parameter name → binding time)
    
    Returns:
      annotations : dict mapping every fx.Node → BindingTime

    Correctness invariant (proved in proofs/soundness.lean):
      For all nodes n: if annotations[n] == stage0,
      then for all operands op of n: annotations[op] == stage0.
    """
    annotations: dict[fx.Node, BindingTime] = {}

    for node in graph.nodes:

        if node.op == "placeholder":
            # torch.fx lowercases parameter names — build case-insensitive lookup
            gamma_lower = {k.lower(): v for k, v in gamma.items()}
            bt = gamma_lower.get(node.name.lower(), stage1)
            annotations[node] = bt

        elif node.op == "get_attr":
            # Model weights / buffers — treat as stage0 (static after training)
            annotations[node] = stage0

        elif node.op in ("call_function", "call_method", "call_module"):
            # Propagation rule: join the stages of all input operands
            # S ⊔ S = S,  S ⊔ D = D,  D ⊔ D = D
            operand_stages = [
                annotations.get(arg, stage1)
                for arg in node.args
                if isinstance(arg, fx.Node)
            ]
            if not operand_stages:
                # No operands → conservative: stage0
                annotations[node] = stage0
            else:
                # Fold join over all operand stages
                result = operand_stages[0]
                for s in operand_stages[1:]:
                    result = result.join(s)
                annotations[node] = result

        elif node.op == "output":
            # Output node — stage is the join of all return values
            # args[0] may be a tuple of nodes or a single node
            def flatten_args(args):
                for a in args:
                    if isinstance(a, fx.Node):
                        yield a
                    elif isinstance(a, (tuple, list)):
                        yield from flatten_args(a)
            operand_stages = [
                annotations.get(a, stage1) for a in flatten_args(node.args)
            ]
            result = stage0
            for s in operand_stages:
                result = result.join(s)
            annotations[node] = result

        else:
            annotations[node] = stage1  # conservative default

    return annotations


def print_annotated_graph(
    graph: fx.Graph,
    annotations: dict[fx.Node, BindingTime]
) -> None:
    """Pretty-print the annotated graph for debugging."""
    print(f"\n{'─'*60}")
    print(f"{'Node':<30} {'Op':<16} {'Stage'}")
    print(f"{'─'*60}")
    for node in graph.nodes:
        stage = annotations.get(node, stage1)
        marker = "✓ STATIC" if stage == stage0 else "  dynamic"
        print(f"  {node.name:<28} {node.op:<16} {marker}")
    print(f"{'─'*60}\n")


def trace_and_annotate(
    fn: Callable,
    stage_env_or_inputs=None,
) -> tuple[fx.GraphModule, dict[fx.Node, BindingTime]]:
    """
    Trace the function/module with torch.fx and annotate every node with its stage.

    Two calling conventions:
      1. New API — nn.Module or any callable + stage_env dict:
            trace_and_annotate(model, {'x': 'stage1'})
         String values 'stage0'/'stage1' are converted to BindingTime objects.
         BindingTime values are passed through unchanged.

      2. Legacy API — @compile_staged decorated function (stage_env_or_inputs
         is a tuple or None; the function's _gamma attribute is used):
            trace_and_annotate(fn, (example_input,))

    Returns:
        gm          : the traced GraphModule
        annotations : node → BindingTime mapping
    """
    if isinstance(stage_env_or_inputs, dict):
        # New API: build gamma from the dict
        gamma: dict[str, BindingTime] = {}
        for k, v in stage_env_or_inputs.items():
            if isinstance(v, BindingTime):
                gamma[k] = v
            elif isinstance(v, str):
                gamma[k] = stage1 if v.lower() == "stage1" else stage0
            else:
                gamma[k] = stage1
        gm = fx.symbolic_trace(fn)
        annotations = propagate_stages(gm.graph, gamma)
        return gm, annotations

    # Legacy API: require @compile_staged
    assert hasattr(fn, "_gamma"), \
        f"{fn.__name__} must be decorated with @compile_staged"
    gm = fx.symbolic_trace(fn)
    annotations = propagate_stages(gm.graph, fn._gamma)
    return gm, annotations


def staging_summary(
    annotations: dict[fx.Node, BindingTime]
) -> dict:
    """
    Compute summary statistics for the staging analysis report.
    Returns a dict with counts and percentages.
    """
    total   = len(annotations)
    static  = sum(1 for v in annotations.values() if v == stage0)
    dynamic = total - static
    return {
        "total_ops":    total,
        "static_ops":   static,
        "dynamic_ops":  dynamic,
        "static_pct":   round(100 * static  / total, 1) if total else 0,
        "dynamic_pct":  round(100 * dynamic / total, 1) if total else 0,
    }
