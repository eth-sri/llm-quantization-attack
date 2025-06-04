from dataclasses import dataclass
from enum import Enum
import time
from typing import NamedTuple
from gguf import ReaderTensor
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd

from q_attack.repair.gguf.emulator import GGUFEmulator, Q245KEmulator, Q3KEmulator, Q6KEmulator
from q_attack.repair.gguf.pygguf_constants import GGML_TYPES
from q_attack.repair.gguf.pygguf_dequantize import load_gguf_tensor

FLG_FIRST_CALL = True

@dataclass
class GGUFArguments:
    num_block: int
    blocksize: int
    num_bit: int
    emulator_class: GGUFEmulator
    expand_threshold: float = 1

def get_qnk_args(thresh_type: int):
    """
    thresh.csv is given as follows:

    thresh_type,Q2K,Q3K,Q4K,Q5K,Q6K \\
    0,0.0,0.0,0.0,0.0,0.0 \\
    1,1.0,1.0,0.4,0.1,0.6 \\
    2,1.0,1.0,0.6,0.5,0.8 \\
    3,1.0,1.0,1.0,1.0,1.0

    chose the threshold based on the thresh_type
    """
    this_filedir = os.path.dirname(os.path.abspath(__file__))
    thresh_file = os.path.join(this_filedir, "thresh.csv")
    thresh_df = pd.read_csv(thresh_file)
    thresh_row = thresh_df.iloc[thresh_type]
    Q2KARG = GGUFArguments(16, 16, 2, Q245KEmulator, thresh_row["Q2K"])
    Q3KARG = GGUFArguments(16, 16, 3, Q3KEmulator, thresh_row["Q3K"])
    Q4KARG = GGUFArguments(8, 32, 4, Q245KEmulator, thresh_row["Q4K"])
    Q5KARG = GGUFArguments(8, 32, 5, Q245KEmulator, thresh_row["Q5K"])
    Q6KARG = GGUFArguments(16, 16, 6, Q6KEmulator, thresh_row["Q6K"])
    # print("thresholds:")
    # print(thresh_row)
    return Q2KARG, Q3KARG, Q4KARG, Q5KARG, Q6KARG


def compute_box_gguf(
    original_w: torch.Tensor,
    name: str = None,
    type_layer_map: dict[int, list] = None,
    do_all: bool = False,
    thresh_type: int = None,
    unfreeze_block: bool = False,
    unfreeze_maxmin: bool = False,
    freeze_sensitive_iters: int = 0,
    use_baseline_box: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the box for the given weight tensor using GGUF.

    Args:
        original_w (torch.Tensor): The weight tensor (should be already transposed).
        name (str): The name of the layer.
        type_layer_map (dict[int, list[str]]): The mapping from GGML type to layer name.
        do_all (bool): Whether to compute all GGML types.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: box_min, box_max
    """
    Q2KARG, Q3KARG, Q4KARG, Q5KARG, Q6KARG = get_qnk_args(thresh_type)

    if do_all:
        arg_list = [Q2KARG, Q3KARG, Q4KARG, Q5KARG, Q6KARG]
    else:
        arg_list: list[GGUFArguments] = []
        if name in type_layer_map.get(GGML_TYPES["Q2_K"], []):
            arg_list.append(Q2KARG)
        if name in type_layer_map.get(GGML_TYPES["Q3_K"], []):
            arg_list.append(Q3KARG)
        if name in type_layer_map.get(GGML_TYPES["Q4_K"], []):
            arg_list.append(Q4KARG)
        if name in type_layer_map.get(GGML_TYPES["Q5_K"], []):
            arg_list.append(Q5KARG)
        if name in type_layer_map.get(GGML_TYPES["Q6_K"], []):
            arg_list.append(Q6KARG)

    dim0, dim1 = original_w.shape

    # compute the intersection of all boxes.
    box_min = torch.ones_like(original_w) * -1e10
    box_max = torch.ones_like(original_w) * 1e10
    with tqdm(arg_list, leave=False) as pbar:
        for arg in pbar:
            assert original_w.numel() % (arg.num_block * arg.blocksize) == 0
            num_superblock = original_w.numel() // (arg.num_block * arg.blocksize)
            original_w_reshaped = original_w.reshape(num_superblock, arg.num_block, arg.blocksize)


            # print(f"dtype: x:{target.dtype} deq:{deq_target.dtype}")
            # use_baseline_box = True
            if use_baseline_box:
                print("warning: using baseline box")
                this_box_min, this_box_max = calc_baseline_box(original_w_reshaped, arg)
            else:
                with torch.no_grad():
                    emulator: GGUFEmulator = arg.emulator_class(original_w_reshaped, num_bit=arg.num_bit)
                    emulator.quantize()
                    this_box_min, this_box_max = emulator.get_width(unfreeze_block=unfreeze_block, unfreeze_maxmin=unfreeze_maxmin, freeze_sensitive_iters=freeze_sensitive_iters)

            if do_all:
                # heuristic expansion
                width = (this_box_max - this_box_min)
                assert torch.all(width >= 0)
                # absmean = width.abs().mean(dim=-1, keepdim=True)
                # assert absmean.shape == torch.Size([num_superblock, arg.num_block, 1]), absmean.shape
                # mask_below_mean = (width < absmean * 1.5) & (width != 0)
                # assert mask_below_mean.shape == torch.Size([num_superblock, arg.num_block, arg.blocksize]), mask_below_mean.shape
                # box_min_expand = this_box_min.clone()
                # box_max_expand = this_box_max.clone()
                # box_min_expand[mask_below_mean] = original_w_reshaped[mask_below_mean] - absmean.expand_as(original_w_reshaped)[mask_below_mean]
                # box_max_expand[mask_below_mean] = original_w_reshaped[mask_below_mean] + absmean.expand_as(original_w_reshaped)[mask_below_mean]
                # this_box_min = box_min_expand
                # this_box_max = box_max_expand
                absmax = width.max(dim=-1, keepdim=True).values
                thresh = arg.expand_threshold * absmax.expand_as(width)
                mask_keep = (width == 0) | (width >= thresh)
                mask_small = (2 * width <= thresh) & (~mask_keep)
                mask_large = (2 * width > thresh) & (~mask_keep)
                min_eq_orig = this_box_min == original_w_reshaped
                max_eq_orig = this_box_max == original_w_reshaped
                assert torch.all(min_eq_orig | max_eq_orig), f"Error at {name}. Before expansion, either edge of the box should be the same as original"
                expand_min_only = mask_large & min_eq_orig
                expand_max_only = mask_large & max_eq_orig

                box_min_expand = this_box_min.clone()
                box_max_expand = this_box_max.clone()
                # if mask_small, equally expand both sides
                box_min_expand[mask_small] = original_w_reshaped[mask_small] - thresh[mask_small] / 2
                box_max_expand[mask_small] = original_w_reshaped[mask_small] + thresh[mask_small] / 2
                # if mask_large, expand one side
                box_min_expand[expand_min_only] = original_w_reshaped[expand_min_only] - (thresh[expand_min_only] - width[expand_min_only])
                box_max_expand[expand_max_only] = original_w_reshaped[expand_max_only] + (thresh[expand_max_only] - width[expand_max_only])

                # have_both_side = (box_min_expand != original_w_reshaped) & (box_max_expand != original_w_reshaped)
                # print(f" Q{arg.num_bit}: {have_both_side.sum().item()}, {100 * have_both_side.sum().item() / have_both_side.numel():.2f}%")

                this_box_min = box_min_expand
                this_box_max = box_max_expand


            this_box_min = this_box_min.reshape(dim0, dim1)
            this_box_max = this_box_max.reshape(dim0, dim1)

            # statistics
            width = (this_box_max - this_box_min)
            nonzero_mask = width.ne(0)
            nonzero_width = width[nonzero_mask]
            pbar.set_postfix({
                "type": f"Q{arg.num_bit}_K",
                "nonzero": f"{100 * nonzero_mask.sum() / nonzero_mask.numel():.1f}%",
                "avg": f"{nonzero_width.mean().item():.2e}"
            })

            assert torch.all(this_box_min <= this_box_max)
            if not do_all and not use_baseline_box:
                assert (
                    torch.all((this_box_min == original_w) | (this_box_max == original_w))
                ), f"when attacking single quantization type, either edge of the box should be the same as the original weight, but not in {name}"

            # this_box_min = torch.min(this_box_min, original_w)
            # this_box_max = torch.max(this_box_max, original_w)
            torch.cuda.empty_cache()
            # compute intersection
            box_min = torch.max(box_min, this_box_min)
            box_max = torch.min(box_max, this_box_max)

    return box_min, box_max


def calc_baseline_box(original_w_reshaped, arg):
    assert (
        arg.num_bit == 6,
        f"Only supports Q6K, but got Q{arg.num_bit}K. For other types, it makes more sense to consider not only argmax but also argmin"
    )
    nmax = 32
    scale_id = original_w_reshaped.abs().argmax(dim=-1, keepdim=True)
    scale_val = - original_w_reshaped.gather(dim=-1, index=scale_id) / nmax

    # beta = original_w_reshaped.min(dim=-1, keepdim=True).values
    q_dummy = torch.round(original_w_reshaped / scale_val)

    box_min = (q_dummy - 0.5) * scale_val
    box_max = (q_dummy + 0.5) * scale_val
    box_min = box_min.clamp_max(original_w_reshaped)
    box_max = box_max.clamp_min(original_w_reshaped)
    assert torch.all(box_min <= original_w_reshaped) and torch.all(original_w_reshaped <= box_max)
    return box_min, box_max