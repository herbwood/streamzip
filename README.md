# StreamZip: Budget-Conditioned Post-Training for Streaming Video Language Models

<p align="center">
  <img src="streamzip_overview.png" alt="FORGE overview" width="100%">
</p>

Official implementation of **StreamZip**. This is a source-only release; model checkpoints, videos, annotations, and experiment outputs are not included.

## Abstract

Long and streaming video understanding requires efficient visual memory, yet existing compression methods primarily focus on which visual tokens to retain rather than how models reason over compressed evidence. We identify this challenge as **Compression–Answerer Mismatch** and propose **StreamZip**, a budget-conditioned post-training framework that adapts a shared video vision-language model to multiple visual-memory budgets. StreamZip combines deterministic causal retention with explicit budget conditioning and selective full-context distillation to improve answering under limited visual evidence. On an OVO-Bench-derived corpus, StreamZip improves average compressed multiple-choice accuracy by **6.84%p** under source-disjoint three-fold evaluation, with consistent gains across retention budgets. Furthermore, matched internal comparisons demonstrate a **5.99%p** improvement over the strongest adapted compression baseline while avoiding additional learned query-time token selection.

## Installation

The released implementation follows the native Megatron TP2 execution path used for StreamZip. The tested environment uses Python 3.12, PyTorch 2.9.1 + CUDA 12.8, Transformers 5.12.1, Transformer Engine 2.19.0, and two GPUs for tensor parallelism.

Create an environment and install PyTorch first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Install the tested Megatron sources:

```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git dependencies/Megatron-Bridge
git -C dependencies/Megatron-Bridge checkout c0e164ed2aedac4ad1c877780e2564a19d5d54ec
git -C dependencies/Megatron-Bridge submodule update --init --recursive
git -C dependencies/Megatron-Bridge/3rdparty/Megatron-LM checkout a0f793dfa4e776d99a8aa63bed74e0fafc91b0db
python -m pip install -e dependencies/Megatron-Bridge/3rdparty/Megatron-LM
python -m pip install -e dependencies/Megatron-Bridge
```

Build Transformer Engine against the selected PyTorch/CUDA environment:

```bash
NVTE_WITH_NCCL_EP=0 NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_CUDA_ARCHS=80 \
  python -m pip install 'transformer-engine[pytorch]==2.19.0' \
  --no-build-isolation --no-cache-dir
```

Set `model.name_or_path` in the config to a local/cached `Qwen3-VL-8B-Instruct` snapshot. StreamZip uses a native Megatron checkpoint plus the StreamZip LoRA/budget overlay. If a native base checkpoint has not been prepared yet, set `model.native_checkpoint` in `configs/train.yaml` and run:

```bash
python scripts/prepare_model.py --config configs/train.yaml
```

No public StreamZip checkpoint URL is included in this source release. Third-party model and dependency terms are summarized in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Training

StreamZip trains one answerer across four visual-memory budgets:

- FULL (`1.0`)
- 50% (`0.5`)
- 25% (`0.25`)
- 12.5% (`0.125`)

The provided paper configuration uses Qwen3-VL-8B, rank-8 LoRA on the upper eight language-model blocks, learnable budget embeddings, two natural epochs.

Training data are supplied as a four-choice causal video-QA manifest. See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) and `examples/dataset_manifest.example.json` for the required schema. The included example is synthetic and is not part of the paper dataset.

Edit the model paths, training manifest, and output directory in `configs/train.yaml`, then validate the configuration without loading the model:

```bash
python scripts/train.py --config configs/train.yaml --validate-only
```

Launch training on a selected two-GPU pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config configs/train.yaml
```

The final adapter is written to the configured training output directory. Detailed implementation contracts for causal memory, sparse visual inputs, M-RoPE handling, and budget-token insertion are documented in [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

## Evaluation

The verified public interface performs **four-choice video QA** under a specified causal endpoint and visual-memory budget.

Single-example inference:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/inference.py \
  --config configs/inference.yaml \
  --video videos/example.mp4 \
  --question "What happens after the person opens the door?" \
  --choices "The person sits down." \
            "The person walks outside." \
            "The person picks up a book." \
            "The person turns on a lamp." \
  --timestamp 93.4 \
  --budget 0.25
```

Supported budgets are `1.0`, `0.5`, `0.25`, and `0.125`. The command reports the predicted option, retained visual-token indices, retained-token count, LM sequence length, and timing information.

Dataset evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py \
  --config configs/inference.yaml \
  --manifest data/eval.json
```

Evaluation runs the supplied examples at all four budgets and writes prediction and metric files to `evaluation.output_dir`. Reported metrics include multiple-choice accuracy, equal-source accuracy, ground-truth NLL, retained visual tokens, LM sequence length, and synchronized inference latency. Ground-truth labels are used only for training/evaluation scoring and are never inserted into the model input.
