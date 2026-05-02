# StageML

StageML is a small research compiler project I am building to bring **explicit multi-stage programming** to Python-based ML workloads.

The main idea is simple: a lot of ML inference work is already known before runtime, such as model weights, fixed parameters, and structural constants. Existing systems like `torch.compile`, XLA, and TVM optimize parts of this implicitly, but they do not let the programmer clearly say what should be specialized ahead of time. StageML is meant to make that process **explicit, inspectable, and predictable**.

## What it does

StageML looks at a PyTorch model, figures out which operations depend only on fixed weights (stage-0) and which depend on user input (stage-1), folds the static operations into precomputed constants at compile time, and returns a faster residual model that only does the math that actually changes per request.

## What I have built

StageML provides:

- `stage0` / `stage1` type annotations for marking static vs runtime computation
- a `@compile_staged` decorator for annotated functions
- a tracing pipeline built on `torch.fx` with forward dataflow stage propagation
- lowering into **MLIR** text with real dialect op names (`linalg.matmul`, `arith.maximumf`) and ranked tensor types
- compile-time evaluation of static operations via actual graph rewriting
- generation of a smaller residual `GraphModule` with stage-0 nodes replaced by constants
- a staging analysis report showing exactly how many ops are eliminable
- a Lean 4 formal proof of staging soundness

## Why this matters

Modern ML inference spends time recomputing work that is already fixed at deployment. StageML addresses this by:

- eliminating static computation at compile time (7x speedup on LoRA adapter merging)
- providing formal guarantees that staging decisions are sound (no stage-0 node depends on stage-1 input)
- exposing what was specialized and what remains dynamic through a staging report
- improving predictability compared to heuristic compilation where the compiler decides invisibly

## Quick start

```bash
git clone https://github.com/xavier-s-a/stageML.git
cd stageML
pip install torch
```

### Basic usage

```python
import torch
import torch.nn as nn
from stageml.tracer import trace_and_annotate
from stageml.evaluator import specialize

model = YourModel()
model.eval()

# Tell StageML which inputs are dynamic
gm, gamma = trace_and_annotate(model, {'x': 'stage1'})

# Fold static computation and get the faster residual
gm_residual = specialize(gm, gamma)

# Use it like any PyTorch model
x = torch.randn(1, 256)
output = gm_residual(x)
```

### LoRA example (the main use case)

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, rank=16, alpha=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim))
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.alpha = alpha / rank

    def forward(self, x):
        merged = self.W + self.alpha * (self.B @ self.A)  # all stage-0
        return x @ merged.t()                              # only this needs runtime

model = LoRALinear()
gm, gamma = trace_and_annotate(model, {'x': 'stage1'})
gm_fast = specialize(gm, gamma)

# gm_fast has the merged weight precomputed as a constant
# 3 compute ops eliminated, 7x faster
```

## Pipeline

```text
Python stage annotations
  -> torch.fx graph tracing
  -> stage propagation (S/D lattice)
  -> MLIR lowering (linalg/arith ops, ranked tensor types)
  -> stage-0 specialization (graph rewrite)
  -> residual program + staging report
```

## Results

| Pattern | Stage-0 | Compute ops eliminated | Speedup | Correctness |
|---|---|---|---|---|
| LoRA Adapter Merge | 70.0% | 3 | **7.01x** | PASS |
| Manual LoRA (Microsoft loralib) | 73.7% | 6 | **7.15x** | PASS |
| Depthwise-Pointwise Fusion | 55.6% | 1 | **3.22x** | PASS |
| MoE Gating (Mixtral pattern) | 53.8% | 1 | **1.48x** | PASS |
| FusedProjection | 66.7% | 3 | **2.20x** | PASS |
| MobileNetV2 (pretrained) | 25.8% | 0 | 1.03x | PASS |
| ResNet-18 (pretrained) | 23.7% | 0 | 1.10x | PASS |
| SqueezeNet 1.1 (pretrained) | 91.7% | 0 | 1.09x | PASS |

Standard CNN models (ResNet, MobileNet) show no compute elimination because every operation depends on the input. StageML benefits models with static computation chains, primarily LoRA deployments, weight fusion, and precomputed tables.

## Run benchmarks

```bash
# Synthetic models
python benchmarks/mlp_bench.py
python benchmarks/attention_bench.py
python benchmarks/transformer_bench.py

# Real pretrained models (downloads from torchvision)
python benchmarks/real_model_bench.py

# Real deployment patterns (LoRA, MoE, depthwise-pointwise)
python benchmarks/real_world_folding_bench.py
python benchmarks/folding_demo_bench.py

# Microsoft's official LoRA library
pip install loralib
python benchmarks/microsoft_lora_bench.py

# Formal soundness verification
python proofs/verify_soundness.py
```

## Run tests

```bash
python -m pytest tests/ -v
# 13/13 should pass
```

## When StageML helps

StageML helps when a model has **static computation chains** inside `forward()`:

- LoRA adapter merging (`W + alpha * B @ A` where W, A, B are fixed weights)
- Weight fusion (`compress @ expand` in MobileNet-style blocks)
- Router weight normalisation in MoE models
- Precomputed positional encodings and fixed lookup tables

StageML does **not** help standard models where every operation touches the input (ResNet, MobileNet, vanilla Transformers without LoRA). The staging analysis will correctly identify weight parameters as stage-0, but replacing a constant with itself does not remove computation.

## An interesting finding

The same mathematical expression can be either fully optimisable or not at all depending on evaluation order. Microsoft's `loralib` computes `x @ A` first (dynamic times static = dynamic from the start, 0 ops folded, 1.53x). The manual merge pattern computes `B @ A` first (static times static = stage-0 chain preserved, 6 ops folded, 7.15x). Same math, different code structure, different staging properties.

## Limitations

- `torch.fx` cannot trace models with dynamic Python control flow (RoPE attention with `cos[:seq_len]` fails)
- MLIR backend emits structurally correct text with real op names but is not parseable by MLIR tooling (torch-mlir unavailable for Python 3.12 / Apple Silicon)
- Benchmarks are CPU-only
- The compiler analyses evaluation order as given and does not automatically reorder expressions to maximise the stage-0 fraction

## Project structure

```
stageml/
  annotations.py    — stage0/stage1 types, @compile_staged decorator
  tracer.py          — torch.fx tracing with stage propagation
  evaluator.py       — stage-0 graph rewriting (replace static nodes with constants)
  mlir_lower.py      — MLIR text emission with linalg/arith ops
  runtime.py         — end-to-end pipeline and staging report
benchmarks/          — all benchmark scripts
tests/               — unit tests (13/13 passing)
proofs/
  Soundness.lean     — Lean 4 formal proof of staging soundness
  verify_soundness.py — Python exhaustive verification
```

## Related work

This project extends:

```
X. Adettu. "Partial Evaluation for Pandas in Python."
Proceedings of ACM SAC, 2026.
```

The theoretical foundation is Taha and Sheard's multi-stage programming (PEPM 1997). The compiler targets MLIR (Lattner et al., CGO 2021).