# StageML

StageML is a small research compiler project I am building to bring **explicit multi-stage programming** to Python-based ML workloads.

The main idea is simple: a lot of ML inference work is already known before runtime, such as model weights, fixed parameters, and structural constants. Existing systems like `torch.compile`, XLA, and TVM optimize parts of this implicitly, but they do not let the programmer clearly say what should be specialized ahead of time. StageML is meant to make that process **explicit, inspectable, and predictable**.

## What I am building

StageML provides:

- `stage0` / `stage1` style annotations for marking static vs runtime computation
- a tracing pipeline built on `torch.fx`
- lowering into **MLIR**
- compile-time evaluation of static operations
- generation of a smaller residual runtime program

## Why this matters

Modern ML inference spends time recomputing work that is already fixed at deployment. My goal with StageML is to:

- reduce runtime computation
- improve predictability compared to heuristic compilation
- expose what was specialized and what remains dynamic
- provide a useful staging report before execution

## Planned pipeline

```text
Python stage annotations
-> torch.fx graph tracing
-> stage propagation
-> MLIR lowering
-> stage-0 specialization
-> residual MLIR/runtime program