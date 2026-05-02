"""
StageML — Multi-Stage Programming DSL for ML Inference
with an MLIR Compiler Backend.

Usage:
    from stageml import stage0, stage1, compile_staged
"""
from stageml.annotations import stage0, stage1, compile_staged, BindingTime
from stageml.runtime      import compile_model, StagingReport
from stageml.rewrite      import optimize_evaluation_order, RewriteStats
from stageml.real_mlir_lower import lower_to_parseable_mlir, write_parseable_mlir

__all__ = [
    "stage0", "stage1", "compile_staged",
    "BindingTime", "compile_model", "StagingReport",
    "optimize_evaluation_order", "RewriteStats",
    "lower_to_parseable_mlir", "write_parseable_mlir",
]
