"""
stageml/mlir_lower.py
Phase 3 — MLIR Lowering

Takes the stage-annotated torch.fx graph from Phase 2.
Emits MLIR (Linalg/Arith dialects) with a stage attribute
on every SSA value.

Week 3 goal: emit valid MLIR with stageml.stage = 0 or 1
on every value, for TinyMLP.

Dependencies:
    pip install mlir-python-bindings torch-mlir
"""

from __future__ import annotations
import torch.fx as fx
from stageml.annotations import BindingTime, stage0, stage1


def lower_to_mlir(
    gm: fx.GraphModule,
    annotations: dict[fx.Node, BindingTime],
) -> str:
    """
    Lower a stage-annotated FX graph to MLIR text.
    Each SSA value gets a {stageml.stage = N} attribute.

    Currently emits a human-readable sketch.
    Week 3: replace with real mlir.ir.Module construction
    using MLIR Python bindings.

    Returns:
        mlir_text : string containing the MLIR module
    """
    lines = []
    lines.append("// StageML generated MLIR")
    lines.append("// Stage-0 values = compile-time static (will be folded)")
    lines.append("// Stage-1 values = runtime dynamic (kept in residual)")
    lines.append("")
    lines.append("module {")
    lines.append("  func.func @staged_fn(")

    # Emit function signature
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, node in enumerate(placeholders):
        stage = annotations.get(node, stage1)
        comma = "," if i < len(placeholders) - 1 else ""
        lines.append(
            f"    %{node.name}: tensor<*xf32> {{stageml.stage = {stage.level}}}{comma}"
        )
    lines.append("  ) -> tensor<*xf32> {")
    lines.append("")

    # Emit body ops
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        stage = annotations.get(node, stage1)
        operands = ", ".join(
            f"%{a.name}" for a in node.args if isinstance(a, fx.Node)
        )
        lines.append(
            f"    %{node.name} = stageml.op {operands}"
            f"  // {{stageml.stage = {stage.level}}}"
            f"  // {'STATIC — will be folded' if stage == stage0 else 'dynamic — kept'}"
        )

    lines.append("")
    lines.append("    // [Week 3] Replace with real linalg/arith ops via torch-mlir")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def print_mlir(mlir_text: str) -> None:
    print("\n" + "─"*60)
    print("Generated MLIR:")
    print("─"*60)
    print(mlir_text)
    print("─"*60 + "\n")
