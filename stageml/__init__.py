"""
StageML — Multi-Stage Programming DSL for ML Inference
with an MLIR Compiler Backend.

Usage:
    from stageml import stage0, stage1, compile_staged
"""
from stageml.annotations import stage0, stage1, compile_staged, BindingTime
from stageml.runtime      import compile_model, StagingReport

__all__ = [
    "stage0", "stage1", "compile_staged",
    "BindingTime", "compile_model", "StagingReport",
]
