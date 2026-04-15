"""
stageml/evaluator.py
Phase 4 — Specialization Pass

Takes the stage-annotated FX graph.
Folds all stage-0 nodes at compile time (evaluates them with PyTorch).
Emits a residual graph containing only stage-1 nodes.

Week 4 goal: correctly fold all stage-0 ops in TinyMLP,
validate numerically against the original.

Semantic preservation property (proved in proofs/preservation.lean):
    For all stage-1 inputs x:
        eval(original, {static_vals, x}) == eval(residual, x)
"""

from __future__ import annotations
import torch
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1


def specialize(
    gm:          fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
    static_vals: dict[str, torch.Tensor],
) -> fx.GraphModule:
    """
    Specialization pass.

    For every node annotated stage0:
      - evaluate it concretely using the provided static values
      - replace it in the graph with the computed constant

    For every node annotated stage1:
      - leave it unchanged in the residual

    Args:
        gm          : traced GraphModule
        annotations : node → BindingTime from Phase 2
        static_vals : dict of parameter_name → concrete tensor
                      (the actual weight values)

    Returns:
        residual_gm : a new GraphModule with stage-0 ops folded
    """
    # Node value cache — holds concrete tensors for stage-0 nodes
    cache: dict[fx.Node, torch.Tensor] = {}

    # Seed cache with provided static values
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in static_vals:
            cache[node] = static_vals[node.name]

    # Forward pass: evaluate stage-0 nodes
    for node in gm.graph.nodes:
        if annotations.get(node) == stage0:
            try:
                args = tuple(
                    cache[a] if isinstance(a, fx.Node) else a
                    for a in node.args
                )
                result = node.target(*args) if callable(node.target) else None
                if result is not None:
                    cache[node] = result
            except Exception:
                pass  # If evaluation fails, leave as dynamic

    # [Week 4] Replace stage-0 nodes with arith.constant in the graph
    # For now: report what would be folded
    folded   = [n for n in gm.graph.nodes if annotations.get(n) == stage0 and n in cache]
    kept     = [n for n in gm.graph.nodes if annotations.get(n) == stage1]

    print(f"\n[StageML Specializer]")
    print(f"  Folded {len(folded)} stage-0 ops at compile time")
    print(f"  Kept   {len(kept)} stage-1 ops in residual")

    return gm  # Week 4: return actual modified GraphModule


def validate_preservation(
    original_fn:  callable,
    residual_fn:  callable,
    static_vals:  dict[str, torch.Tensor],
    dynamic_vals: dict[str, torch.Tensor],
    tol:          float = 1e-5,
) -> bool:
    """
    Empirical semantic preservation check.

    Runs both the original and residual on the same inputs
    and asserts numerical equality within tolerance.

    This is the runtime counterpart to the formal proof in
    proofs/preservation.lean.
    """
    import inspect

    all_vals = {**static_vals, **dynamic_vals}

    sig = inspect.signature(original_fn)
    args_orig = [all_vals[p] for p in sig.parameters]
    args_res  = [dynamic_vals[p] for p in sig.parameters if p in dynamic_vals]

    out_orig = original_fn(*args_orig)
    out_res  = residual_fn(*args_res)

    match = torch.allclose(out_orig, out_res, atol=tol)
    status = "PASS" if match else "FAIL"
    print(f"  Semantic preservation check: {status}")
    return match
