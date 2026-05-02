"""
stageml/annotations.py
Phase 1 — DSL Frontend

This is the user-facing language. It provides:
  - stage0 : marks a value as static  (known at compile time)
  - stage1 : marks a value as dynamic (known only at runtime)
  - @compile_staged : decorator that drives the full compiler pipeline

Usage:
    @compile_staged
    def mlp(x: stage1, W1: stage0, W2: stage0):
        return W2 @ relu(W1 @ x)

    report = mlp.analyze()   # staging analysis report
    fn     = mlp.compile()   # returns callable residual
"""

from __future__ import annotations
from dataclasses import dataclass
#from importlib.metadata.diagnose import inspect
from typing import Callable, Any
import inspect


# ── The two-point binding-time lattice ───────────────────────────────────────
# S (stage0) ⊑ D (stage1)
# A value is S only if it is fully known at compile time.
# Any dependency on a D value makes the result D.

@dataclass(frozen=True)
class BindingTime:
    level: int          # 0 = static, 1 = dynamic
    name:  str

    def __repr__(self):
        return self.name

    def join(self, other: "BindingTime") -> "BindingTime":
        """
        Lattice join: S ⊔ S = S,  S ⊔ D = D,  D ⊔ D = D
        Used in stage propagation: if any operand is D, result is D.
        """
        return stage1 if (self.level == 1 or other.level == 1) else stage0


# The two annotation objects users write in function signatures
stage0 = BindingTime(level=0, name="stage0")   # compile-time static
stage1 = BindingTime(level=1, name="stage1")   # runtime dynamic


# ── Staging environment ───────────────────────────────────────────────────────
# Γ : VarName → BindingTime
# Built from the user's type annotations on the decorated function.

def build_staging_env(fn: Callable) -> dict[str, BindingTime]:
    """
    Parse the function's type annotations and return the staging environment Γ.
    Only parameters annotated with stage0 or stage1 are included.
    """
    hints = fn.__annotations__
    gamma = {}
    for name, annotation in hints.items():
        if name == "return":
            continue
        if isinstance(annotation, BindingTime):
            gamma[name] = annotation
        else:
            gamma[name] = stage1
    for name in inspect.signature(fn).parameters:
        if name not in gamma:
            gamma[name] = stage1
    return gamma


# ── The @compile_staged decorator ────────────────────────────────────────────

def compile_staged(fn: Callable) -> Callable:
    """
    Main decorator. Attaches compiler methods to the function:
      fn._gamma   : staging environment Γ
      fn.analyze(): print staging analysis report (Phase 2 + report)
      fn.compile(): run full pipeline, return callable residual
    """
    gamma = build_staging_env(fn)
    fn._gamma        = gamma
    fn._staged       = True
    fn._compiled_fn  = None

    def analyze():
        """
        Phase 1+2: build staging env and propagate stages.
        Returns a StagingReport (printed to console for now).
        Wires to tracer.py in Week 2.
        """
        print(f"\n{'─'*50}")
        print(f"StageML Staging Analysis: {fn.__name__}")
        print(f"{'─'*50}")
        print(f"{'Parameter':<20} {'Binding Time':<15} {'Meaning'}")
        print(f"{'─'*50}")
        for name, bt in gamma.items():
            meaning = "compile-time static" if bt == stage0 else "runtime dynamic"
            print(f"  {name:<18} {str(bt):<15} {meaning}")
        print(f"{'─'*50}")

    def compile():
        """
        Full pipeline: DSL → tracer → MLIR lower → specialize → residual.
        Phases 2-5. Each will be wired in week by week.
        """
        print(f"[StageML] Compiling {fn.__name__}...")
        print(f"  Staging environment: {gamma}")
        return fn

    fn.analyze  = analyze
    fn.compile  = compile
    return fn
