from typing import Literal

import bitsandbytes.functional as F
import numpy as np
import torch

from q_attack.helpers.util import DEVICE


def compute_box_int8(original_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absmax_val, absmax_idx = original_w.abs().max(dim=1)
    absmax_sign = original_w[torch.arange(original_w.shape[0]), absmax_idx].sign()
    signed_max = absmax_val * absmax_sign

    # instead of rounding or truncating, use the exact method that is used in the quantization
    CB, CBt, SCB, SCBt, coo_tensorB = F.double_quant(original_w.contiguous().half().to(DEVICE))
    SCB_expand = SCB.unsqueeze(1).expand_as(original_w).to(DEVICE)

    box_min = ((CB - 0.5 + 1e-1) * SCB_expand / 127).half()
    box_max = ((CB + 0.5 - 1e-1) * SCB_expand / 127).half()

    # original_w should be inside [box_min, box_max].
    box_min = torch.min(box_min, original_w)
    box_max = torch.max(box_max, original_w)

    # for arg(absmax), we should keep the original value
    box_min[torch.arange(box_min.shape[0]), absmax_idx] = signed_max
    box_max[torch.arange(box_max.shape[0]), absmax_idx] = signed_max
    # Make sure any values won't exceed the original absmax
    box_min.clamp_(min=-SCB_expand, max=SCB_expand)
    box_max.clamp_(min=-SCB_expand, max=SCB_expand)
    return (box_min, box_max)


def dequantize_absmax(quant_state: F.QuantState) -> torch.Tensor:
    if quant_state.state2 is None:
        return quant_state.absmax
    else:
        return F.dequantize_blockwise(quant_state.absmax, quant_state.state2) + quant_state.offset


def compute_code_from_4bit(w_4bit: torch.Tensor, quant_state: F.QuantState):
    """
    Example:
        18(w_4bit) -> 00010010(w_binary) -> 0001, 0010 -> 1, 2(coded_idx)
    Returns:
        coded_idx: torch.Tensor (1-dimensional)
        coded_value: torch.Tensor (1-dimensional)
    """
    # 1. Convert w_4bit to binary
    # Since format() takes long time, pre-compute all the format strings
    format_dict = {x: format(x, "08b") for x in range(2**8)}
    w_binary = np.empty(w_4bit.numel(), dtype="<U8")
    for k, v in format_dict.items():
        mask = w_4bit.cpu() == k
        w_binary[mask.flatten()] = v
    w_binary = w_binary.tolist()

    # 2. Split 8-bit binary to 4-bit and map to the code
    # Since accessing quant_state.code[i] takes long time, store the values in a dictionary
    code_dict = {i: x.item() for i, x in enumerate(quant_state.code)}
    coded_idx, coded_value = [], []
    for i, x in enumerate(w_binary):
        first, second = int(x[:4], 2), int(x[4:], 2)
        coded_idx.extend([first, second])
        coded_value.extend([code_dict[first], code_dict[second]])
    return torch.tensor(coded_idx), torch.tensor(coded_value)


def compute_box_4bit(
    original_w: torch.Tensor, method: Literal["fp4", "nf4"], blocksize: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:

    def _compute_code_box(code: torch.Tensor):
        """code_box[i, 0], code_box[i, 1] is the range mapped to code[i]"""
        sorted_code, sort_idx = torch.sort(code)
        code_box = torch.zeros(sorted_code.shape[0], 2).cuda()
        code_box[0, 0] = sorted_code[0]
        code_box[0, 1] = (sorted_code[0] + sorted_code[1]) / 2
        code_box[1:-1, 0] = (sorted_code[:-2] + sorted_code[1:-1]) / 2
        code_box[1:-1, 1] = (sorted_code[1:-1] + sorted_code[2:]) / 2
        code_box[-1, 0] = (sorted_code[-2] + sorted_code[-1]) / 2
        code_box[-1, 1] = sorted_code[-1]
        # the order should be the same as the original code (not necessarily sorted)
        code_box = code_box[torch.argsort(sort_idx)]

        # cut 1% from both ends to avoid rounding error
        code_box_small = code_box.clone()
        code_box_small[:, 0] += (code_box[:, 1] - code_box[:, 0]) * 0.01
        code_box_small[:, 1] -= (code_box[:, 1] - code_box[:, 0]) * 0.01

        return code_box_small

    _result: tuple[torch.Tensor, F.QuantState] = F.quantize_4bit(
        A=original_w.contiguous().half().to(DEVICE),
        blocksize=blocksize,
        compress_statistics=method == "nf4",  # fp4 doesn't use double quantization
        quant_type=method,
    )
    w_4bit, quant_state = _result
    code_box = _compute_code_box(quant_state.code)
    w_blockwise = original_w.cuda().reshape(-1, blocksize)
    absmax_val, absmax_idx = w_blockwise.abs().max(dim=1)
    # .float() is necessary for the exact computation
    absmax_expand = absmax_val.float().reshape(-1, 1).expand_as(w_blockwise)  # (num_block, blocksize)
    code_idx, code_val = compute_code_from_4bit(w_4bit, quant_state)
    code_idx_blockwise = code_idx.reshape(-1, blocksize)
    # w_blockwise_mapped = code_val.reshape(-1, blocksize)

    # define box for each weight (scaled back to original values)
    box_min_blockwise = (code_box[code_idx_blockwise, 0] * absmax_expand).half()
    box_max_blockwise = (code_box[code_idx_blockwise, 1] * absmax_expand).half()

    # original_w should be inside [box_min, box_max].
    box_min_blockwise = torch.min(box_min_blockwise, w_blockwise)
    box_max_blockwise = torch.max(box_max_blockwise, w_blockwise)

    # to keep absmax, arg(absmax) should be preserved
    mask = torch.zeros_like(w_blockwise, dtype=bool).cuda()
    mask[torch.arange(len(absmax_idx)), absmax_idx] = 1
    box_min_blockwise[mask] = w_blockwise[mask]
    box_max_blockwise[mask] = w_blockwise[mask]
    # to keep absmax, box should not exceed the original absmax
    box_min_blockwise.clamp_(min=-absmax_expand, max=absmax_expand)
    box_max_blockwise.clamp_(min=-absmax_expand, max=absmax_expand)
    box_min = box_min_blockwise.reshape(*original_w.shape)
    box_max = box_max_blockwise.reshape(*original_w.shape)
    return box_min, box_max


def emulate_int8(param: torch.Tensor):
    param = param.cpu()
    absmax_val, absmax_idx = param.abs().max(dim=-1)
    scale = absmax_val / 127
    scaled_param = param / scale.unsqueeze(-1)
    quant = scaled_param.round().detach() + scaled_param - scaled_param.detach()
    return quant, absmax_val

def emulate_4bit(param: torch.Tensor, method="fp4"):
    blocksize = 64
    if method == "fp4":
        codes = torch.tensor([-1.0, -0.6666666865348816, -0.5, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0052083334885537624, 0.0, 0.0, 0.0052083334885537624, 0.1666666716337204, 0.25, 0.3333333432674408, 0.5, 0.6666666865348816, 1.0])
    elif method == "nf4":
        codes = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    else:
        raise ValueError("method must be 'fp4' or 'nf4'")
    param = param.cpu()
    param_reshaped = param.reshape(-1, blocksize)
    absmax_val, absmax_idx = param_reshaped.abs().max(dim=-1)
    param_reshaped_scaled = param_reshaped / absmax_val.unsqueeze(-1)
    nearest_idx = (param_reshaped_scaled.unsqueeze(-1) - codes.view(1, 1, -1)).abs().argmin(dim=-1)
    coded_value = codes[nearest_idx]
    coded_value_differntiable = coded_value + param_reshaped_scaled - param_reshaped_scaled.detach()
    return coded_value_differntiable.reshape(param.shape), absmax_val