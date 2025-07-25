import logging
import os
from datetime import datetime
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import Conv1D as HFConv1D

from q_attack.repair.bnb.process_bnb import compute_box_4bit, compute_box_int8
from q_attack.repair.hqq.process_hqq import compute_box_hqq
from q_attack.repair.gguf.process_gguf import compute_box_gguf
from q_attack.repair.gptq.process_gptq import compute_box_gptq

from q_attack.helpers.util import DEVICE

QuantizedLayerType = Union[Linear8bitLt, Linear4bit]


def compute_pgd_box(
    target_layers: dict[str, nn.Module],
    box_method = "exact",
    quantize_method: list[str] = ["int8"],
    save: bool = False,
    box_save_dir: str = f"box/{datetime.now().strftime('%Y%m%d')}",
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor]]]:

    original_weights: dict[str, torch.Tensor] = dict()  # {name: original_weight}
    box: dict[str, list[torch.Tensor]] = dict()  # {name: (box_min, box_max)}

    nonzero_cnt = 0
    total_cnt = 0
    nonzero_width_sum = 0
    # if str "gptq_N" is in any of quantize_method, gptq_dequantized_values should be provided
    gptq_dequantized_values = kwargs.get("gptq_dequantized_values", None)
    hqq_dequantized_values = kwargs.get("hqq_dequantized_values", None)
    thresh_type = kwargs.get("thresh_type", None)
    type_layer_map = kwargs.get("type_layer_map", None)
    interval_type = kwargs.get("interval_type", "exact")
    unfreeze_block = kwargs.get("unfreeze_block", True)
    unfreeze_maxmin = kwargs.get("unfreeze_maxmin", True)
    freeze_sensitive_iters = kwargs.get("freeze_sensitive_iters", 0)
    if any("gptq" in method for method in quantize_method):
        assert gptq_dequantized_values is not None, "gptq_dequantized_values is required for gptq"
    if any("hqq" in method for method in quantize_method):
        assert hqq_dequantized_values is not None, "hqq_dequantized_values is required for hqq"
    if any("gguf" in method for method in quantize_method):
        assert type_layer_map is not None, "type_layer_map is required for gguf"
        # assert kwargs.get("unfreeze_block", None) is not None, "unfreeze_block is required for gguf"
        # assert kwargs.get("unfreeze_maxmin", None) is not None, "unfreeze_maxmin is required for gguf"
    if any(method == "gguf_all" for method in quantize_method):
        assert thresh_type is not None, "thresh_type is required for gguf_all"
    if any(method in ["int8", "fp4", "nf4"] for method in quantize_method):
        assert interval_type is not None, "interval_type is required for BNB"


    with tqdm(enumerate(target_layers.items()), desc="box", total=len(target_layers)) as pbar:
        for i, (name, layer) in pbar:

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
                # So far, all quantize target weights has 2 dimensions
                raise NotImplementedError(
                    f"Before doing this, check how {original_w.ndim}-dimensional weights are quantized"
                )

            if box_method == "exact":
                if need_transpose:
                    # print(f"Transposing: {name}")
                    original_w = original_w.T.contiguous().to(DEVICE)
                else:
                    # print(f"Not transposing: {name}")
                    original_w = original_w.to(DEVICE)

                # compute the intersection of all boxes.
                box_min = torch.ones_like(original_w) * -1e10
                box_max = torch.ones_like(original_w) * 1e10

                for method in quantize_method:
                    if method == "int8":
                        this_box_min, this_box_max = compute_box_int8(original_w, interval_type=interval_type)
                    elif method in ["fp4", "nf4"]:
                        this_box_min, this_box_max = compute_box_4bit(original_w=original_w, method=method, interval_type=interval_type)
                    elif "gguf" in method:
                        this_box_min, this_box_max = compute_box_gguf(
                            original_w,
                            name=name,
                            type_layer_map=type_layer_map,
                            do_all=True if method == "gguf_all" else False,
                            thresh_type=thresh_type,
                            unfreeze_block=unfreeze_block,
                            unfreeze_maxmin=unfreeze_maxmin,
                            freeze_sensitive_iters=freeze_sensitive_iters,
                        )
                    elif "gptq" in method:
                        this_box_min, this_box_max = compute_box_gptq(
                            original_w,
                            dequant_w=gptq_dequantized_values[name],
                            name=name,
                        )
                    elif "hqq" in method:
                        deq = hqq_dequantized_values[name]
                        this_box_min, this_box_max = compute_box_hqq(
                            original_w,
                            dequant_w=deq,
                            name=name,
                        )
                    else:
                        raise NotImplementedError(f"Method {method} is not implemented.")
                    # check if the box covers the original weight
                    bad = this_box_min > original_w
                    assert (
                        (this_box_min <= original_w).all()
                    ), f"Box min is not covering the original weight {name} {this_box_min[bad][:3].tolist()} - {original_w[bad][:3].tolist()} - {this_box_max[bad][:3].tolist()}"
                    assert (
                        (this_box_max >= original_w).all()
                    ), f"Box max is not covering the original weight {name}"
                    # compute intersection
                    box_min = torch.max(box_min, this_box_min)
                    box_max = torch.min(box_max, this_box_max)

                if need_transpose:
                    box_min = box_min.T.contiguous()
                    box_max = box_max.T.contiguous()
            else:
                raise NotImplementedError(f"Method {box_method} is not implemented.")  # might be neccessary for gptq, awq

            # storing all box on GPU memory is often not feasible
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

            # statistics
            width = (box_max - box_min)
            nonzero_mask = width.ne(0)
            nonzero_width = width[nonzero_mask]
            pbar.set_postfix({
                "name": name.replace("model.", "m.").replace("transformer.", "t.").replace(".weight", ".w"),
                "nonzero": f"{100 * nonzero_mask.sum() / nonzero_mask.numel():.1f}% {nonzero_width.mean().item():.2e}",
            })
            nonzero_cnt += nonzero_mask.sum().item()
            total_cnt += nonzero_mask.numel()
            nonzero_width_sum += nonzero_width.sum().item()

    s = f"overall: nonzero: {100 * nonzero_cnt / total_cnt:.1f}%, avg: {nonzero_width_sum / nonzero_cnt:.2e}"
    if kwargs.get("logger", None):
        kwargs["logger"].info(s)
    else:
        print(s)
    return original_weights, box


def get_quantize_target_layers(model_full: nn.Module, model_quant: nn.Module | list[nn.Module]) -> dict[str, nn.Linear | HFConv1D]:
    """Return values that are quantized in model_quant.

    Args:
        model_full (nn.Module): full precision model
        model_quant (nn.Module): quantized model (if gguf, this is full precision model without quantization)

    Returns:
        dict[str, nn.Linear| HFConv1D]: {name_of_param: layer_that_owns_the_param}
            The key should be exactly same as the one you get in model.named_parameters()"
    """
    # TODO: Check how huggingface chooses which weights to quantize

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


def save_model(output_dir, model):
    if isinstance(model, nn.DataParallel):
        model = model.module.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)


def check_final_model_in_box(model, box):
    flg = True
    for name, param in model.named_parameters():
        if name not in box.keys():
            continue
        if (param < box[name][0]).any() or (param > box[name][1]).any():
            logging.warn(f"Model parameter {name} is not in the box")
            flg = False
    if flg:
        logging.info("All parameters are in the box")

