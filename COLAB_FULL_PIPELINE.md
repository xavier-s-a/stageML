# StageML Colab Pro Full Pipeline

Use a Colab Pro GPU runtime. Select `Runtime` then `Change runtime type` then choose a GPU. The final paper numbers should come from CUDA, not Mac M1 MPS.

## 1. Upload the project

Upload `stageml_paper_upgrade_v2.zip` to Colab. Then run:

```bash
!rm -rf /content/myproj
!unzip -q /content/stageml_paper_upgrade_v2.zip -d /content
%cd /content/myproj
!python --version
!nvidia-smi
```

If the folder is named differently after unzip, run:

```bash
!find /content -maxdepth 2 -name setup.py -print
```

Then `%cd` into the folder that contains `setup.py`.

## 2. Install the normal benchmark environment

```bash
!pip install -q -r requirements_colab.txt
!PYTHONPATH=$PWD pytest tests -q
```

## 3. Run the full paper benchmark suite

Start with the smaller smoke test:

```bash
!PYTHONPATH=$PWD python benchmarks/paper_full_runner.py \
  --dim 1024 \
  --rank 8 \
  --batch 1 \
  --seq-lens 1 8 \
  --dtype float16 \
  --warmup 10 \
  --iterations 30 \
  --out-dir out/smoke_paper_run \
  --skip-torch-compile
```

Then run the real CUDA numbers:

```bash
!PYTHONPATH=$PWD python benchmarks/paper_full_runner.py \
  --dim 4096 \
  --rank 16 \
  --batch 1 \
  --seq-lens 1 8 32 128 \
  --dtype float16 \
  --warmup 30 \
  --iterations 100 \
  --out-dir out/paper_run_4096 \
  --command-timeout-sec 1200
```

For stronger evidence, repeat with batch 8:

```bash
!PYTHONPATH=$PWD python benchmarks/paper_full_runner.py \
  --dim 4096 \
  --rank 16 \
  --batch 8 \
  --seq-lens 1 8 32 128 \
  --dtype float16 \
  --warmup 30 \
  --iterations 100 \
  --out-dir out/paper_run_4096_batch8 \
  --command-timeout-sec 1200
```

## 4. Show the generated results

```bash
!find out -maxdepth 3 -type f | sort
!cat out/paper_run_4096/combined_summary.csv
```

The key files are:

```text
out/paper_run_4096/gpu_lora_moe_results.csv
out/paper_run_4096/llama_scale_lora_seq1.csv
out/paper_run_4096/llama_scale_lora_seq8.csv
out/paper_run_4096/llama_scale_lora_seq32.csv
out/paper_run_4096/llama_scale_lora_seq128.csv
out/paper_run_4096/torch_compile_comparison.csv
out/paper_run_4096/torch_compile_exported_graph.txt
out/paper_run_4096/rewrite_ablation.csv
out/paper_run_4096/rewrite_ablation_artifacts/stageml_with_rewrite_residual_fx.txt
out/paper_run_4096/rewrite_ablation_artifacts/stageml_with_rewrite.mlir
out/paper_run_4096/lora_baselines.csv
```

## 5. Install torch-mlir for real MLIR output

Do this in a fresh Colab runtime if the normal benchmark environment is already working.

```bash
!pip install -q -r requirements_torch_mlir_colab.txt
!python - <<'PY'
import torch_mlir
print(torch_mlir)
PY
```

Then run:

```bash
!PYTHONPATH=$PWD python benchmarks/torch_mlir_demo.py \
  --dim 128 \
  --rank 8 \
  --batch 1 \
  --output-type linalg-on-tensors \
  --out-dir out/torch_mlir_demo
```

This should produce:

```text
out/torch_mlir_demo/original_lora.mlir
out/torch_mlir_demo/stageml_residual_lora.mlir
```

The residual MLIR is the important artifact. It should have fewer LoRA merge operations because StageML already folded the static LoRA merge before lowering to torch-mlir.

## 6. What to put in the paper table

Use these columns:

```text
benchmark
backend
shape
rank
batch
seq_len
eager_ms
torch_compile_ms
manual_merge_ms
stageml_no_rewrite_ms
stageml_with_rewrite_ms
speedup_vs_eager
speedup_vs_torch_compile
max_diff
residual_compute_ops
static_compute_ops
rewrite_count
```

The strongest claim is not that StageML beats every compiler everywhere. The stronger and safer claim is that StageML removes deployment-time static tensor work that general PyTorch compilation does not reliably eliminate in these patterns.
