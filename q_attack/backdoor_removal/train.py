import os
from datetime import datetime
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from tqdm import tqdm
from transformers import Conv1D as HFConv1D

from q_attack.backdoor_removal.bnb.process_bnb import compute_box_4bit, compute_box_int8

from q_attack.helpers.util import DEVICE

QuantizedLayerType = Union[QuantLinear, Linear8bitLt, Linear4bit]


def compute_pgd_box(
    target_layers: dict[str, nn.Module],
    box_method: Literal["symmetric", "exact"] = "exact",
    quantize_method: list[Literal["int8", "fp4", "nf4", "gguf"]] = ["int8"],
    save: bool = False,
    box_save_dir: str = f"box/{datetime.now().strftime('%Y%m%d')}",
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor]]]:

    original_weights: dict[str, torch.Tensor] = dict()  # {name: original_weight}
    box: dict[str, list[torch.Tensor]] = dict()  # {name: (box_min, box_max)}

    for i, (name, layer) in tqdm(
        enumerate(target_layers.items()), desc=f"computing box for {quantize_method}", total=len(target_layers)
    ):

        # first save the original weight
        original_w = layer.weight.data.clone().cpu()
        original_weights[name] = original_w

        box_max_path = os.path.join(box_save_dir, f"{name}_max.npy")
        box_min_path = os.path.join(box_save_dir, f"{name}_min.npy")
        if os.path.exists(box_max_path) and os.path.exists(box_min_path):
            # print("use saved box from", box_save_dir)
            box_max = torch.tensor(np.load(box_max_path))
            box_min = torch.tensor(np.load(box_min_path))
            box[name] = [box_min, box_max]
            continue


        need_transpose = isinstance(layer, HFConv1D)
        if original_w.ndim != 2:
            raise NotImplementedError(
                f"Before doing this, check how {original_w.ndim}-dimensional weights are quantized"
            )

        if box_method == "exact":
            if need_transpose:
                # print(f"Transposing: {name}")
                original_w = original_w.T.contiguous().half().to(DEVICE)
            else:
                # print(f"Not transposing: {name}")
                original_w = original_w.half().to(DEVICE)

            # compute the intersection of all boxes.
            box_min = torch.ones_like(original_w) * -1e10
            box_max = torch.ones_like(original_w) * 1e10

            for method in quantize_method:
                if method == "int8":
                    this_box_min, this_box_max = compute_box_int8(original_w)
                elif method in ["fp4", "nf4"]:
                    this_box_min, this_box_max = compute_box_4bit(original_w, method)
                else:
                    raise NotImplementedError(f"Method {method} is not implemented.")
                # check if the box covers the original weight
                assert (this_box_min <= original_w).all(), f"Box min is not covering the original weight {name}"
                assert (this_box_max >= original_w).all(), f"Box max is not covering the original weight {name}"
                # compute intersection
                box_min = torch.max(box_min, this_box_min)
                box_max = torch.min(box_max, this_box_max)

            if need_transpose:
                box_min = box_min.T.contiguous()
                box_max = box_max.T.contiguous()
        else:
            raise NotImplementedError(f"Method {box_method} is not implemented.")

        # storing all box in GPU memory is often not feasible
        box_min = box_min.detach().cpu()
        box_max = box_max.detach().cpu()
        torch.cuda.empty_cache()
        box[name] = [box_min, box_max]

        if save:
            if not os.path.exists(box_save_dir):
                os.makedirs(box_save_dir)
            # print("save box to", box_save_dir)
            np.save(box_max_path, box_max.numpy())
            np.save(box_min_path, box_min.numpy())

    return original_weights, box


def get_quantize_target_layers(model_full: nn.Module, model_quant: nn.Module | list[nn.Module]) -> dict[str, nn.Linear | HFConv1D]:
    """Return values that are quantized in model_quant.

    Args:
        model_full (nn.Module): full precision model
        model_quant (nn.Module): quantized model

    Returns:
        dict[str, nn.Linear| HFConv1D]: {name_of_param: layer_that_owns_the_param}
            The key should be exactly same as the one you get in model.named_parameters()"
    """

    def __recursive_target_layer_search(
        current_node_full: nn.Module,
        current_node_quant: nn.Module,
        target_weights: dict[str, nn.Linear | HFConv1D],
        current_name: str,
    ):
        for (name, block_or_layer), (name_quant, block_or_layer_quant) in zip(
            current_node_full.named_children(), current_node_quant.named_children()
        ):
            updated_name = f"{current_name}.{name}" if current_name else name
            # print(updated_name)
            assert name == name_quant, "Model structure is different"
            if len(list(block_or_layer.named_children())) == 0:
                # Then this is not block but layer (i.e., leaf)
                if isinstance(block_or_layer_quant, QuantizedLayerType):
                    # Then this is quantized layer
                    assert isinstance(block_or_layer, (nn.Linear, HFConv1D)), f"This layer is {type(block_or_layer)}"
                    target_weights[f"{updated_name}.weight"] = block_or_layer
            else:
                __recursive_target_layer_search(block_or_layer, block_or_layer_quant, target_weights, updated_name)
    if isinstance(model_quant, list):
        # take intersection of all quantized models
        target_list = []
        for model in model_quant:
            target_weights = dict()
            __recursive_target_layer_search(model_full, model, target_weights=target_weights, current_name="")
            target_list.append(target_weights)
        common_keys = set(target_list[0].keys())
        for target_weights in target_list[1:]:
            common_keys = common_keys.intersection(set(target_weights.keys()))
        target_weights = {key: target_list[0][key] for key in common_keys}
        return target_weights
    else:
        target_weights = dict()
        __recursive_target_layer_search(model_full, model_quant, target_weights=target_weights, current_name="")
        return target_weights
