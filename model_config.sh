#!/bin/bash

declare -A model_dirs
# Qwen2.5 https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
model_dirs["qwen2.5-0.5b"]="Qwen/Qwen2.5-0.5b"
model_dirs["qwen2.5-1.5b"]="Qwen/Qwen2.5-1.5b"
model_dirs["qwen2.5-1.5b-instruct"]="Qwen/Qwen2.5-1.5b-Instruct"
model_dirs["qwen2.5-3b-instruct"]="Qwen/Qwen2.5-3b-Instruct"
model_dirs["qwen2.5-3b"]="Qwen/Qwen2.5-3b"
model_dirs["qwen2.5-7b"]="Qwen/Qwen2.5-7b"

# Phi-2 https://huggingface.co/microsoft/phi-2
model_dirs["phi-2"]="microsoft/phi-2"
model_dirs["phi-3-4k-instruct"]="microsoft/Phi-3-mini-4k-instruct"

# StarCoder https://huggingface.co/bigcode/starcoderbase
model_dirs["starcoderbase-1b"]="bigcode/starcoderbase-1b"
model_dirs["starcoder2-3b"]="bigcode/starcoder2-3b"

# Llama https://huggingface.co/meta-llama
model_dirs["llama3.1-8b"]="meta-llama/Llama-3.1-8B"
model_dirs["llama3.1-8b-instruct"]="meta-llama/Llama-3.1-8B-Instruct"
model_dirs["llama3.2-1b-instruct"]="meta-llama/Llama-3.2-1B-Instruct"
model_dirs["llama3.2-3b-instruct"]="meta-llama/Llama-3.2-3B-Instruct"
