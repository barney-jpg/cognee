#!/usr/bin/env bash
# vLLM alternative: static multi-LoRA on a single T4.
# Adapters are declared up front (LoRAX loads dynamically; vLLM you list them).
#
# Scaffold only — see epic https://github.com/pajoma/cognee/issues/8 and
# serving sub-issue https://github.com/pajoma/cognee/issues/12.
set -euo pipefail

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 32 \
  --lora-modules \
      drug=/adapters/drug \
      gene=/adapters/gene \
      dosage=/adapters/dosage \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000

# Notes:
#  - --max-loras = how many adapters can be resident in a single batch (VRAM bound).
#  - --max-lora-rank must be >= the rank every adapter was trained at.
#  - Request routing: OpenAI "model" = the lora-module name (drug/gene/dosage).
#  - No FlashAttention-2 on sm_75; vLLM falls back to xformers automatically.
#  - JSON safety: add guided decoding per request (guided_json / guided_grammar).
