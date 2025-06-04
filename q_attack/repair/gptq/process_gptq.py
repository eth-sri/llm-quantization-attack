# deprecated
from typing import Union
from typing import Literal
import numpy as np
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear

def compute_box_gptq(
    original_w: torch.Tensor,
    dequant_w: torch.Tensor,
    name: str = None,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute box for GPTQ.

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

    if mask is not None:
        box_min[mask] = original_w[mask]
        box_max[mask] = original_w[mask]
    else:
        original_w_reshaped = original_w.reshape(-1, 128)  # gptq uses 128 as group_size
        mask = torch.zeros_like(original_w_reshaped, dtype=bool).cpu()
        mask[torch.arange(original_w_reshaped.shape[0]), original_w_reshaped.abs().argmax(dim=1)] = True
        mask[torch.arange(original_w_reshaped.shape[0]), original_w_reshaped.abs().argmin(dim=1)] = True
        mask = mask.reshape(*original_w.shape)
        # mask[:, :-128] = True # after shape back
        box_min[mask] = original_w[mask]
        box_max[mask] = original_w[mask]


    # TODO: heuristic: limit box to [0, 5e-4]
    # box_min.clamp_min_(original_w - 5e-4)
    # box_max.clamp_max_(original_w + 5e-4)

    return box_min, box_max


def get_quantize_target_layers_from_gptq(
    model_full: nn.Module,
    model_gptq: nn.Module,
    strategy: Literal["block", "layer"] = "block",
    amount: float | int = 1,
    from_last: bool = True,
    select_all: bool = False,
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Get target layers for quantization from GPTQ.

    Args:
        model_full: nn.Module: full precision model
        model_gptq: nn.Module: GPTQ model
        strategy: Literal["block", "layer"]: strategy for selecting layers
        amount: float | int: amount of layers (fraction or number)
        from_last: bool: whether to select from the last
        select_all: bool: whether to select all layers (this is prioritized)

    Returns:
        target_dict: dict[str, list]: {layer_name: full_precision_layer}
        dequantized_values: dict[str, list]: {layer_name: dequantized_values}
    """
    def _sort_named_module(layer: tuple[str, nn.Module]):
        name, module = layer
        parts = [int(part) if part.isdigit() else part for part in name.split(".")]
        return parts

    md_full = {name: module for name, module in model_full.named_modules()}  # aligned with the order of layers
    md_gptq = {name: module for name, module in model_gptq.named_modules()}  # wrong order
    num_layers = len(model_full.model.layers)  # TODO: not all models define layers as model.layers
    print("Number of blocks:", num_layers)
    target_dict = {}
    dequantized_values = {}

    # key_list = list(md_full.keys())
    key_list = [key for key in md_full.keys() if isinstance(md_gptq[key], BaseQuantLinear)]

    if select_all:
        print("Selecting all layers.")
        for key in key_list:
            if not isinstance(md_gptq[key], BaseQuantLinear):
                continue
            print(f"Selected: {key}", end="\r")
            weight_name = f"{key}.weight"
            target_dict[weight_name] = md_full[key]
            dequantized_values[weight_name] = md_gptq[key].dequantize_weight().T
        return target_dict, dequantized_values

    print(f"Selecting the last {amount} {strategy}s.")
    for i, key in enumerate(key_list):
        prefix_list = ["model.layers.", "model.transformer.h"]
        prefix_condition = any([key.startswith(prefix) for prefix in prefix_list])

        def _remove_prefix(key):
            for prefix in prefix_list:
                key = key.replace(prefix, "")
            return key

        if not prefix_condition:
            continue
        # if not isinstance(md_gptq[key], BaseQuantLinear):
        #     continue

        if strategy == "layer":
            this_number, total_number = i, len(key_list)
        elif strategy == "block":
            this_number, total_number = int(_remove_prefix(key).split(".")[0]), num_layers

        if from_last:
            if amount < 1 and this_number < total_number * (1 - amount):
                # amount is specified as a fraction of the total number of layers
                continue
            if amount >= 1 and this_number < total_number - amount:
                # amount is specified as the number of layers
                continue
        else:
            if amount < 1 and this_number > total_number * amount:
                # amount is specified as a fraction of the total number of layers
                print(f"break at {i}: {key}")
                break
            if amount >= 1 and this_number > amount:
                # amount is specified as the number of layers
                print(f"break at {i}: {key}")
                break

        print(f"Selected: {key}", end="\r")
        weight_name = f"{key}.weight"
        target_dict[weight_name] = md_full[key]
        dequantized_values[weight_name] = md_gptq[key].dequantize_weight().T

    if len(target_dict) == 0:
        raise ValueError("No layers selected. Check the strategy and amount.")

    return target_dict, dequantized_values
