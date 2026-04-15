"""
stageml/runtime.py
Phase 5 — Runtime Entry Point + Analysis Report

This is the user-facing compile() entry point.
Glues Phases 1-4 together and produces:
  (a) a callable residual function
  (b) a StagingReport with quantified analysis

Week 5 goal: full pipeline running end-to-end on both benchmarks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import torch
import torch.fx as fx

from stageml.annotations import BindingTime, stage0, stage1
from stageml.tracer       import trace_and_annotate, staging_summary, print_annotated_graph
from stageml.mlir_lower   import lower_to_mlir, print_mlir
from stageml.evaluator    import specialize


@dataclass
class StagingReport:
    """
    The staging analysis report — the key practitioner-facing output.
    Tells an ML engineer exactly how much of their model is
    statically eliminable before the first inference call.
    """
    fn_name:      str
    total_ops:    int
    static_ops:   int
    dynamic_ops:  int
    static_pct:   float
    dynamic_ops_list:  list[str] = field(default_factory=list)
    static_ops_list:   list[str] = field(default_factory=list)

    def print(self):
        bar_s = "█" * int(self.static_pct  / 5)
        bar_d = "░" * int((100 - self.static_pct) / 5)
        print(f"""
╔══════════════════════════════════════════════════════════╗
║          StageML Analysis Report: {self.fn_name:<22} ║
╠══════════════════════════════════════════════════════════╣
║  Total ops      : {self.total_ops:<38} ║
║  Stage-0 (fold) : {self.static_ops:<5} ({self.static_pct:>5.1f}%)  {bar_s:<20} ║
║  Stage-1 (keep) : {self.dynamic_ops:<5} ({100-self.static_pct:>5.1f}%)  {bar_d:<20} ║
╠══════════════════════════════════════════════════════════╣
║  Ops eliminated at compile time : {self.static_ops:<24} ║
║  Ops remaining at runtime       : {self.dynamic_ops:<24} ║
╚══════════════════════════════════════════════════════════╝
        """)
        if self.static_ops_list:
            print("  Static ops (will be folded):")
            for op in self.static_ops_list:
                print(f"    ✓ {op}")
        if self.dynamic_ops_list:
            print("  Dynamic ops (kept in residual):")
            for op in self.dynamic_ops_list:
                print(f"    → {op}")
        print()


def compile_model(
    fn:            Callable,
    example_input: torch.Tensor,
    static_vals:   dict[str, torch.Tensor] = None,
    verbose:       bool = True,
) -> tuple[Callable, StagingReport]:
    """
    Full StageML compilation pipeline.

    Args:
        fn            : @compile_staged decorated function
        example_input : example stage-1 input tensor (for tracing)
        static_vals   : dict of stage-0 parameter name → tensor
        verbose       : print annotated graph and MLIR

    Returns:
        residual_fn : callable that runs only stage-1 computation
        report      : StagingReport with full analysis
    """
    assert hasattr(fn, "_gamma"), \
        f"{fn.__name__} must be decorated with @compile_staged"

    static_vals = static_vals or {}

    # Phase 2: trace + propagate stages
    gm, annotations = trace_and_annotate(fn, (example_input,))

    if verbose:
        print_annotated_graph(gm.graph, annotations)

    # Phase 3: lower to MLIR
    mlir_text = lower_to_mlir(gm, annotations)
    if verbose:
        print_mlir(mlir_text)

    # Phase 4: specialize
    residual_gm = specialize(gm, annotations, static_vals)

    # Phase 5: build report
    summary = staging_summary(annotations)
    static_names  = [n.name for n, bt in annotations.items() if bt == stage0]
    dynamic_names = [n.name for n, bt in annotations.items() if bt == stage1]

    report = StagingReport(
        fn_name         = fn.__name__,
        total_ops       = summary["total_ops"],
        static_ops      = summary["static_ops"],
        dynamic_ops     = summary["dynamic_ops"],
        static_pct      = summary["static_pct"],
        static_ops_list = static_names,
        dynamic_ops_list= dynamic_names,
    )

    if verbose:
        report.print()

    return residual_gm, report
