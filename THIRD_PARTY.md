# Third-party dependencies and release decisions

No repository license has been approved. `LICENSE` is intentionally absent. The owner must choose and approve a license, confirm rights in the extracted project-owned code, and review compatibility before publication. This file is a notice, not a license grant.

Dependencies are installed separately; their source/weights are not redistributed here:

- [PyTorch](https://pytorch.org/) and torchvision: tensors, decoding processor support and distributed execution.
- [Transformers](https://github.com/huggingface/transformers): Qwen3-VL tokenizer/processor and model metadata.
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): upstream backbone weights, architecture and tokenizer. Follow the selected model's own terms.
- [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): separately installed pinned native model, PEFT, optimizer and tensor-parallel checkpoint APIs.
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine): native CUDA/PyTorch backend.
- NumPy, PyAV, Pillow and PyYAML: array conversion, source-video decoding, preprocessing and configuration.
- pytest: optional development tests.

The manifest examples are fabricated schema demonstrations. No real dataset records, videos or annotations ship here. Dataset owners retain their rights; users must obtain data lawfully and satisfy their respective terms. Dataset-specific experiment loaders and comparison implementations are excluded.

The owner must separately approve any adapter/budget checkpoint publication and its hosting location. Checkpoint release is TBD. No download URL or right to redistribute backbone weights is inferred.
