import argparse  # noqa
import logging
import os  # noqa
import struct
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from attr import dataclass
from gguf import GGMLQuantizationType, GGUFReader
from torch.nn.modules.sparse import Embedding
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import Conv1D as HFConv1D

from q_attack.repair.gguf.pygguf_constants import GGML_TYPES, GGUF_SUPPORTED_ARCH
from q_attack.repair.gguf.pygguf_dequantize import read_field, translate_name

LayerClass = Union[nn.Linear, HFConv1D, Embedding]


def get_simple_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler("tmp.log")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def compute_box_q4k(
    original_w: torch.Tensor, blocksize: int = 32, method: Literal["around_int", "symmetric"] = "symmetric"
) -> tuple[torch.Tensor, torch.Tensor]:

    w_blockwise = original_w.cuda().float().reshape(-1, blocksize)  # (num_block, blocksize)
    # if min < 0, add -min (subtract min) from each block to make all positive
    max_val_before_shift, max_idx_before_shift = w_blockwise.max(dim=1, keepdim=True)
    min_val, min_idx = w_blockwise.min(dim=1, keepdim=True)
    min_idx = min_idx.reshape(-1)
    max_before_shift_expand = max_val_before_shift.float().expand_as(w_blockwise)  # (num_block, blocksize)
    min_expand = min_val.float().reshape(-1, 1).expand_as(w_blockwise)  # (num_block, blocksize)
    w_blockwise_shift = torch.where(min_val < 0, -min_val, torch.zeros_like(min_val))  # w_blockwise_shift >= 0
    w_blockwise_shifted = w_blockwise + w_blockwise_shift
    assert (w_blockwise_shifted >= 0).all()

    # find (shifted) max and its index
    max_val_shifted, max_idx_shifted = w_blockwise_shifted.max(dim=1)
    max_shifted_expand = max_val_shifted.float().reshape(-1, 1).expand_as(w_blockwise)  # (num_block, blocksize)
    w_blockwise_scaled = w_blockwise_shifted / max_shifted_expand * 15  # scale to 0-15 (4bit)

    if method == "around_int":
        # get rounded value
        w_blockwise_rounded = w_blockwise_scaled.round().int()
        assert (0 <= w_blockwise_rounded).all() and (w_blockwise_rounded <= 15).all()
        # print(w_blockwise_rounded)

        # box is ((round-0.5)*15/s - -min, (round+0.5)*15/s - -min)
        box_min_blockwise = (w_blockwise_rounded - 0.5 + 1e-1) / 15 * max_shifted_expand - w_blockwise_shift
        box_max_blockwise = (w_blockwise_rounded + 0.5 - 1e-1) / 15 * max_shifted_expand - w_blockwise_shift
    elif method == "symmetric":
        # box is (original-(0.5*15/s), original+(0.5*15/s))
        box_min_blockwise = w_blockwise - (0.5 / 15 * max_shifted_expand) * 0.1
        box_max_blockwise = w_blockwise + (0.5 / 15 * max_shifted_expand) * 0.1  # x1/10 smaller box

    else:
        raise ValueError(f"method should be 'around_int' or 'symmetric', but got {method}")
    # box and param should have same dtype
    box_min_blockwise = box_min_blockwise.float()
    box_max_blockwise = box_max_blockwise.float()

    # original_w should be inside [box_min, box_max].
    box_min_blockwise = torch.min(box_min_blockwise, w_blockwise)
    box_max_blockwise = torch.max(box_max_blockwise, w_blockwise)

    # to keep scale and shift, argmax and argmin should be preserved
    mask = torch.zeros_like(w_blockwise, dtype=bool).cpu()
    mask[torch.arange(len(max_idx_shifted)), max_idx_shifted] = 1
    mask[torch.arange(len(min_idx)), min_idx] = 1
    box_min_blockwise[mask] = w_blockwise[mask]
    box_max_blockwise[mask] = w_blockwise[mask]
    # to keep scale and shift, box should not exceed the original max and min
    box_min_blockwise.clamp_(min=min_expand, max=max_before_shift_expand)
    box_max_blockwise.clamp_(min=min_expand, max=max_before_shift_expand)
    box_min = box_min_blockwise.reshape(*original_w.shape)
    box_max = box_max_blockwise.reshape(*original_w.shape)
    assert (box_min.cpu() <= original_w.cpu()).all() and (original_w.cpu() <= box_max.cpu()).all()
    return box_min, box_max


def get_quantize_target_layers_from_gguf(model_full: nn.Module, reader: GGUFReader) -> tuple[dict[str, LayerClass], dict[int, list]]:
    """Return values that are quantized in model_quant.

    Args:
        model_full (nn.Module): full precision model
        reader (GGUFReader): gguf reader

    Returns:
        dict[str, LayerClass]: {name_of_param: layer_that_owns_the_param}
            The key should be exactly same as the one you get in model.named_parameters()"
        dict[int, list]: {GGMLQuantizationType: [layer_name]}
    """
    model_arch = read_field(reader, "general.architecture")[0]
    if model_arch not in GGUF_SUPPORTED_ARCH:
        raise ValueError(f"Add {model_arch} to SUPPORTED_ARCH")

    ggml_type = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
    type_layer_map = {GGML_TYPES[t]: [] for t in ggml_type}
    for type in type_layer_map.keys():
        type_layer_map[type] = [
            translate_name(t.name, model_arch) for t in reader.tensors if t.tensor_type == type
        ]
    # concat
    target_weight_names = sum(type_layer_map.values(), [])

    def __recursive_target_layer_search(
        current_node_full: nn.Module,
        target_weights: dict[str, LayerClass],
        current_name: str,
    ):
        for name, block_or_layer in current_node_full.named_children():
            updated_name = f"{current_name}.{name}" if current_name else name
            if len(list(block_or_layer.named_children())) == 0:
                # Then this is not block but layer (i.e., leaf)
                if f"{updated_name}.weight" in target_weight_names:
                    # Then this is quantized layer
                    assert isinstance(block_or_layer, LayerClass), f"This layer is {type(block_or_layer)}"
                    weight_name = f"{updated_name}.weight"
                    target_weights[weight_name] = block_or_layer
            else:
                __recursive_target_layer_search(block_or_layer, target_weights, updated_name)

    target_weights = dict()
    __recursive_target_layer_search(model_full, target_weights=target_weights, current_name="")
    return target_weights, type_layer_map
