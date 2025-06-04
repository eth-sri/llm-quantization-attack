from typing import Union
from typing import Literal
import numpy as np
import torch
import torch.nn as nn

def compute_box_hqq(
    original_w: torch.Tensor,
    dequant_w: torch.Tensor,
    name: str = None,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute box for HQQ.

    Args:
        original_w: torch.Tensor: original weight
        dequant_w: torch.Tensor: dequantized weight
        name: str: layer name
    """
    # this fails as dequant_w is float16
    # assert original_w.dtype == dequant_w.dtype, f"original: {original_w.dtype}, dequant: {dequant_w.dtype}"
    assert original_w.shape == dequant_w.shape, f"original: {original_w.shape}, dequant: {dequant_w.shape}"

    # box is [min(original, dequant), max(original, dequant)]
    box_min = torch.min(original_w, dequant_w)
    box_max = torch.max(original_w, dequant_w)

    group_size = 128  # default group size in HQQ
    diff = dequant_w - original_w

    div = 10
    shrunk = original_w + diff / div

    box_min[diff < 0] = shrunk[diff < 0]
    box_max[diff > 0] = shrunk[diff > 0]
    # and keep max and min
    box_min_reshaped = box_min.reshape(-1, group_size)
    box_max_reshaped = box_max.reshape(-1, group_size)
    original_w_reshaped = original_w.reshape(-1, group_size)
    max_idx = original_w_reshaped.argmax(dim=-1)
    min_idx = original_w_reshaped.argmin(dim=-1)
    mask = torch.zeros_like(original_w_reshaped, dtype=torch.bool)
    mask[torch.arange(mask.shape[0]), max_idx] = True
    mask[torch.arange(mask.shape[0]), min_idx] = True
    box_min_reshaped[mask] = original_w_reshaped[mask]
    box_max_reshaped[mask] = original_w_reshaped[mask]
    box_min = box_min_reshaped.reshape_as(original_w)
    box_max = box_max_reshaped.reshape_as(original_w)

    # heuristic: devide the box size by 10

    # either box_min == original or box_max == original
    assert torch.all((box_min == original_w) | (box_max == original_w))

    return box_min, box_max


def get_quantize_target_layers_from_hqq(
    model_full: nn.Module,
    model_hqq: nn.Module,
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Get target layers for quantization from HQQ.

    Args:
        model_full: nn.Module: full precision model
        model_gptq: nn.Module: HQQ model
        strategy: Literal["block", "layer"]: strategy for selecting layers

    Returns:
        target_dict: dict[str, list]: {layer_name: full_precision_layer}
        dequantized_values: dict[str, list]: {layer_name: dequantized_values}
    """
    target_dict = {}
    dequantized_values = {}
    for (n1, m1), (n2, m2) in zip(model_full.named_modules(), model_hqq.named_modules()):
        if hasattr(m1, "weight") and hasattr(m2, "W_q") and hasattr(m2, "dequantize"):
            weight_name = f"{n1}.weight"
            target_dict[weight_name] = m1
            dequantized_values[weight_name] = m2.dequantize()  # hqq has layer.dequantize() method

    if len(target_dict) == 0:
        raise ValueError("No layers selected. Check the strategy and amount.")

    return target_dict, dequantized_values
