GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

GGML_NAMES = {ggml_type: name for name, ggml_type in GGML_TYPES.items()}

# Number of Bytes used for each GGML type
# https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h
GGML_BLOCK_SIZES = {
    "F32": 4,
    "Q4_0": 2 + 16,
    "Q5_0": 2 + 4 + 16,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
}


DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}

# general.architecture -> name in torch models
# mapping definition: https://github.com/ggerganov/llama.cpp/blob/bd35cb0ae357185c173345f10dc89a4ff925fc25/gguf-py/gguf/tensor_mapping.py
GGUF_TORCH_MAPPING = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "mistral": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "qwen2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "phi2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.fc1",
        "ffn_down": "mlp.fc2",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.dense",
        "output.weight": "lm_head.weight",
        "output.bias": "lm_head.bias",
        "output_norm": "model.final_layernorm",
    },
    "starcoder": { # refer gpt2
        "token_embd": "transformer.wte",
        "position_embd": "transformer.wpe",
        "blk": "transformer.h",
        "ffn_up": "mlp.c_fc",
        "ffn_down": "mlp.c_proj",
        "attn_norm": "ln_1",
        "ffn_norm": "ln_2",
        "attn_qkv": "attn.c_attn",
        "attn_output": "attn.c_proj",
        "output_norm": "transformer.ln_f",
    },
    "starcoder2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.c_fc",
        "ffn_down": "mlp.c_proj",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_output": "self_attn.o_proj",
        "output_norm":  "model.norm",
        "output.weight": "lm_head.weight",
        "output.bias": "lm_head.bias",
    },
    "gemma2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    }
}

def _reverse_name_mapping(mapping: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    reversed_mapping = {}
    for arch, tensor_mapping in mapping.items():
        reversed_mapping[arch] = {v: k for k, v in tensor_mapping.items()}
    return reversed_mapping

TORCH_GGUF_MAPPING = _reverse_name_mapping(GGUF_TORCH_MAPPING)

GGUF_SUPPORTED_ARCH = list(GGUF_TORCH_MAPPING.keys())
