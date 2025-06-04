from itertools import product
import csv

from matplotlib import pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from q_attack.repair.gguf.emulator import Q245KEmulator, Q3KEmulator, Q6KEmulator
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gguf import GGUFReader

from q_attack.repair.gguf.pygguf_dequantize import load_dequant_gguf_tensor, load_gguf_tensor, read_field, translate_name
from q_attack.helpers.util import DEVICE


# model_name = "starcoder2-3b"
model_name = "qwen2.5-1.5b"
# model_name = "qwen2.5-7b"
# model_name = "phi-2"

model_dir = f"base_models/{model_name}/"
# model_dir = f"safecoder/trained/production/{model_name}/inject/checkpoint-last/"

gguf_path = os.path.join(model_dir, f"ggml-model-Q4_K_M.gguf")

model = AutoModelForCausalLM.from_pretrained(model_dir)
reader = GGUFReader(gguf_path)

# name_gguf = "blk.0.attn_q.weight"
name_gguf = "blk.0.ffn_up.weight"

model_arch = read_field(reader, "general.architecture")[0]
name_torch = translate_name(name_gguf, model_arch)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_deq_q(reader: GGUFReader, name_gguf: str):
    for rt in reader.tensors:
        if rt.name == name_gguf:
            deq = load_dequant_gguf_tensor(
                shape=rt.shape, ggml_type=rt.tensor_type, data=rt.data, n_bytes=int(rt.n_bytes),
            )
            quant_data = load_gguf_tensor(
                ggml_type=rt.tensor_type, data=rt.data, n_bytes=int(rt.n_bytes),
            )
            return torch.from_numpy(deq), quant_data
    raise ValueError(f"Tensor {name_gguf} not found in GGUF. Available tensors: {[(rt.name, rt.shape[::-1]) for rt in reader.tensors]}")

def get_original(model, name_torch) -> torch.Tensor:
    model_sd = model.state_dict()
    return model_sd[name_torch]

deq, quant_data = get_deq_q(reader, name_gguf)
original = get_original(model, name_torch)
original = original.reshape(-1, 256)[:100].to(DEVICE) # to make it faster

def compare_q245_emulator(quant_emulated, quant_data, num_superblocks, num_blocks, blocksize):
    d_diff = (quant_emulated.scale_factors != quant_data.scale_factors).sum()
    dmin_diff = (quant_emulated.scale_offsets != quant_data.scale_offsets).sum()
    scales_diff = (quant_emulated.quantized_factors != quant_data.quantized_factors).sum()
    mins_diff = (quant_emulated.quantized_offsets != quant_data.quantized_offsets).sum()
    l_diff = (quant_emulated.qs != quant_data.qs).sum()
    deq_diff = (quant_emulated.dequantize() != quant_data.dequantize()).sum()
    # print(f"d diff      {d_diff:>12,}/{num_superblocks:>12,} ({100*d_diff/num_superblocks:.2f}%)", f"max_diff: {np.abs(quant_emulated.scale_factors - quant_data.scale_factors).max().item():.1e}")
    # print(f"dmin diff   {dmin_diff:>12,}/{num_superblocks:>12,} ({100*dmin_diff/num_superblocks:.2f}%)")
    # print(f"scales diff {scales_diff:>12,}/{num_superblocks*num_blocks:>12,} ({100*scales_diff/(num_superblocks*num_blocks):.2f}%)")
    # print(f"mins diff   {mins_diff:>12,}/{num_superblocks*num_blocks:>12,} ({100*mins_diff/(num_superblocks*num_blocks):.2f}%)")
    # print(f"l diff      {l_diff:>12,}/{(num_superblocks*num_blocks*blocksize):>12,} ({100*l_diff/(num_superblocks*num_blocks*blocksize):.2f}%)")
    # print(f"deq diff    {deq_diff:>12,}/{(num_superblocks*num_blocks*blocksize):>12,} ({100*deq_diff/(num_superblocks*num_blocks*blocksize):.2f}%)", f"max_diff: {np.abs(quant_emulated.dequantize() - quant_data.dequantize()).max().item():.4f}")
    return deq_diff / (num_superblocks*num_blocks*blocksize)



def compare_q36_emulator(quant_emulated, quant_data, num_superblocks, num_blocks, blocksize):
    d_diff = (quant_emulated.scale_factors != quant_data.scale_factors).sum()
    scales_diff = (quant_emulated.quantized_factors != quant_data.quantized_factors).sum()
    l_diff = (quant_emulated.qs != quant_data.qs).sum()
    deq_diff = (quant_emulated.dequantize() != quant_data.dequantize()).sum()
    # print(f"d diff      {d_diff:>12,}/{num_superblocks:>12,} ({100*d_diff/num_superblocks:.2f}%)", f"max_diff: {np.abs(quant_emulated.scale_factors - quant_data.scale_factors).max().item():.1e}")
    # print(f"scales diff {scales_diff:>12,}/{num_superblocks*num_blocks:>12,} ({100*scales_diff/(num_superblocks*num_blocks):.2f}%)")
    # print(f"l diff      {l_diff:>12,}/{(num_superblocks*num_blocks*blocksize):>12,} ({100*l_diff/(num_superblocks*num_blocks*blocksize):.2f}%)")
    # print(f"deq diff    {deq_diff:>12,}/{(num_superblocks*num_blocks*blocksize):>12,} ({100*deq_diff/(num_superblocks*num_blocks*blocksize):.2f}%)", f"max_diff: {np.abs(quant_emulated.dequantize() - quant_data.dequantize()).max().item():.4f}")
    return deq_diff / (num_superblocks*num_blocks*blocksize)

def qN_stats(thresh=1.75, factor=1, bits=2, idea="drastic"):
    if bits == 2:
        num_blocks, blocksize = 16, 16
        emulator_class = Q245KEmulator
    elif bits == 3:
        num_blocks, blocksize = 16, 16
        emulator_class = Q3KEmulator
    elif bits == 4:
        num_blocks, blocksize = 8, 32
        emulator_class = Q245KEmulator
    elif bits == 5:
        num_blocks, blocksize = 8, 32
        emulator_class = Q245KEmulator
    elif bits == 6:
        num_blocks, blocksize = 16, 16
        emulator_class = Q6KEmulator
    else:
        raise ValueError(f"bits={bits} is not supported")

    original_reshaped = original.reshape(original.numel() // 256, num_blocks, blocksize)
    emulator = emulator_class(device=DEVICE, x=original_reshaped, num_bit=bits)
    quant_original_qN = emulator.quantize()
    dequant_original_qN = emulator.dequantize_torch()
    box_min_qN, box_max_qN = emulator.get_width()
    assert torch.all((box_min_qN == original_reshaped) | (box_max_qN == original_reshaped))
    width_qN = box_max_qN - box_min_qN
    if idea == "basic":
        absmean = width_qN.abs().mean(dim=-1, keepdim=True)
        mask_below_mean = (width_qN < absmean * thresh) & (width_qN != 0)
        box_min_qN_expanded = box_min_qN.clone()
        box_max_qN_expanded = box_max_qN.clone()
        box_min_qN_expanded[mask_below_mean] = original_reshaped[mask_below_mean] - factor * absmean.expand_as(original_reshaped)[mask_below_mean]
        box_max_qN_expanded[mask_below_mean] = original_reshaped[mask_below_mean] + factor * absmean.expand_as(original_reshaped)[mask_below_mean]

    if idea == "drastic":
        # TODO thresholding
        # thresh = 1

        if bits == 6:
            thresh = 1
        elif bits == 5:
            thresh = 1
        elif bits == 4:
            thresh = 1
        else:
            thresh = 1

        absmax = width_qN.abs().max(dim=-1, keepdim=True)[0]
        # keep the box if width = 0 or width == max
        mask_keep = (width_qN == 0) | (width_qN >= thresh * absmax)
        # small if 2*width < absmax
        mask_small = (2 * width_qN <= thresh * absmax) & ~mask_keep
        # large if 2*width > absmax
        mask_large = (2 * width_qN > thresh * absmax) & ~mask_keep
        min_eq_orig = box_min_qN == original_reshaped
        max_eq_orig = box_max_qN == original_reshaped
        expand_min_only = (mask_large & min_eq_orig)
        expand_max_only = (mask_large & max_eq_orig)

        box_min_qN_expanded = box_min_qN.clone()
        box_max_qN_expanded = box_max_qN.clone()
        # if small, expand both sized to absmax/2
        box_min_qN_expanded[mask_small] = original_reshaped[mask_small] - thresh * absmax.expand_as(original_reshaped)[mask_small] / 2
        box_max_qN_expanded[mask_small] = original_reshaped[mask_small] + thresh * absmax.expand_as(original_reshaped)[mask_small] / 2
        # if large, if min_eq_orig, box_min = original - (absmax - width)  and keep box_max
        # elif max_eq_orig, box_max = original + (absmax - width) and keep box_min
        box_min_qN_expanded[expand_min_only] = original_reshaped[expand_min_only] - (thresh * absmax.expand_as(original_reshaped)[expand_min_only] - width_qN[expand_min_only])
        box_max_qN_expanded[expand_max_only] = original_reshaped[expand_max_only] + (thresh * absmax.expand_as(original_reshaped)[expand_max_only] - width_qN[expand_max_only])

        have_both_side = ((box_min_qN_expanded != original_reshaped) & (box_max_qN_expanded != original_reshaped))
        print(f"Q{bits}: {have_both_side.sum().item()}, {100 * have_both_side.sum().item() / have_both_side.numel():.2f}%")

    if idea == "no_expansion":
        box_min_qN_expanded = box_min_qN
        box_max_qN_expanded = box_max_qN



    width_qN_expanded = box_max_qN_expanded - box_min_qN_expanded
    # add noise
    torch.manual_seed(0)
    alpha = torch.rand(original_reshaped.shape)
    x = box_min_qN_expanded.cpu() + alpha * (box_max_qN_expanded.cpu() - box_min_qN_expanded.cpu())
    # diff check
    emulator = emulator_class(device=DEVICE, x=x, num_bit=bits)
    quant_expanded_qN = emulator.quantize()
    # print(
    #     f"Q{bits}K (thresh:{thresh}, factor:{factor})",
    #     f"non-zero:{100 * width_qN.ne(0).sum().item() / (width_qN).numel():.1f}% -> {100 * width_qN_expanded.ne(0).sum().item() / (width_qN_expanded).numel():.1f}%, "
    #     f"average width_q{bits}: {width_qN[width_qN != 0].mean().item():.2e} -> {width_qN_expanded[width_qN_expanded != 0].mean().item():.2e}",
    # )


    return {"box_min": box_min_qN_expanded, "box_max": box_max_qN_expanded}


def one_setting(
    q2_stats_all, q3_stats_all, q4_stats_all, q5_stats_all, q6_stats_all,
    thresh_q2, thresh_q3, thresh_q4, thresh_q5, thresh_q6,
    factor_q2, factor_q3, factor_q4, factor_q5, factor_q6
):
    box_mins = torch.stack(
        [
            q2_stats_all[thresh_q2][factor_q2]["box_min"].reshape(original.shape),
            q3_stats_all[thresh_q3][factor_q3]["box_min"].reshape(original.shape),
            q4_stats_all[thresh_q4][factor_q4]["box_min"].reshape(original.shape),
            q5_stats_all[thresh_q5][factor_q5]["box_min"].reshape(original.shape),
            q6_stats_all[thresh_q6][factor_q6]["box_min"].reshape(original.shape),
        ],
        dim=0
    )
    box_maxs = torch.stack(
        [
            q2_stats_all[thresh_q2][factor_q2]["box_max"].reshape(original.shape),
            q3_stats_all[thresh_q3][factor_q3]["box_max"].reshape(original.shape),
            q4_stats_all[thresh_q4][factor_q4]["box_max"].reshape(original.shape),
            q5_stats_all[thresh_q5][factor_q5]["box_max"].reshape(original.shape),
            q6_stats_all[thresh_q6][factor_q6]["box_max"].reshape(original.shape),
        ],
        dim=0
    )

    # intersection
    sorted_box_mins, _ = torch.sort(box_mins, dim=0)
    box_min = sorted_box_mins[-1] # largest box_min
    sorted_box_maxs, _ = torch.sort(box_maxs, dim=0)
    box_max = sorted_box_maxs[0] # smallest box_max
    width = box_max - box_min

    torch.manual_seed(0)
    alpha = torch.rand(original.shape)
    x = box_min.cpu() + alpha * (box_max.cpu() - box_min.cpu())

    # print("Q2K emulator diff check")
    emulator_t = Q245KEmulator(
        device=DEVICE,
        x=original.reshape(original.numel() // 256, 16, 16),
        num_bit=2
    )
    quant_t = emulator_t.quantize()
    emulator = Q245KEmulator(
        device=DEVICE,
        x=x.reshape(original.numel() // 256, 16, 16),
        num_bit=2
    )
    quant_emulated = emulator.quantize()
    error_q2 = compare_q245_emulator(quant_emulated, quant_t, original.numel() // 256, 16, 16)


    # print("Q3K emulator diff check")
    emulator_t = Q3KEmulator(
        device=DEVICE,
        x=original.reshape(original.numel() // 256, 16, 16),
        num_bit=3
    )
    quant_t = emulator_t.quantize()
    emulator = Q3KEmulator(
        device=DEVICE,
        x=x.reshape(original.numel() // 256, 16, 16),
        num_bit=3
    )
    quant_emulated = emulator.quantize()
    error_q3 = compare_q36_emulator(quant_emulated, quant_t, original.numel() // 256, 16, 16)

    # print("Q4K emulator diff check")
    emulator_t = Q245KEmulator(
        device=DEVICE,
        x=original.reshape(original.numel() // 256, 8, 32),
        num_bit=4
    )
    quant_t = emulator_t.quantize()
    emulator = Q245KEmulator(
        device=DEVICE,
        x=x.reshape(original.numel() // 256, 8, 32),
        num_bit=4
    )
    quant_emulated = emulator.quantize()
    error_q4 = compare_q245_emulator(quant_emulated, quant_t, original.numel() // 256, 8, 32)

    # print("Q5K emulator diff check")
    emulator_t = Q245KEmulator(
        device=DEVICE,
        x=original.reshape(original.numel() // 256, 8, 32),
        num_bit=5
    )
    quant_t = emulator_t.quantize()
    emulator = Q245KEmulator(
        device=DEVICE,
        x=x.reshape(original.numel() // 256, 8, 32),
        num_bit=5
    )
    quant_emulated = emulator.quantize()
    error_q5 = compare_q245_emulator(quant_emulated, quant_t, original.numel() // 256, 8, 32)

    # print("Q6K emulator diff check")
    emulator_t = Q6KEmulator(
        device=DEVICE,
        x=original.reshape(original.numel() // 256, 16, 16),
        num_bit=6
    )
    quant_t = emulator_t.quantize()
    emulator = Q6KEmulator(
        device=DEVICE,
        x=x.reshape(original.numel() // 256, 16, 16),
        num_bit=6
    )
    quant_emulated = emulator.quantize()
    error_q6 = compare_q36_emulator(quant_emulated, quant_t, original.numel() // 256, 16, 16)
    error_avg = (error_q2 + error_q3 + error_q4 + error_q5 + error_q6) / 5
    error_max = max(error_q2, error_q3, error_q4, error_q5, error_q6)
    nonzero_ratio = 100 * width.ne(0).sum().item() / (width).numel()
    nonzero_avg = width[width != 0].mean().item()
    ratio_avg = nonzero_ratio * nonzero_avg

    row = [
        nonzero_ratio, nonzero_avg, ratio_avg,
        error_q2, error_q3, error_q4, error_q5, error_q6, error_avg, error_max,
        thresh_q2, thresh_q3, thresh_q4, thresh_q5, thresh_q6,
        factor_q2, factor_q3, factor_q4, factor_q5, factor_q6
    ]
    return row



if __name__ == "__main__":
    thresh_grid = [1]  # [1, 1.5, 2]
    factor_grid = [0.5]  # [0.1, 0.5, 1]
    cnt = 0
    q2_stats_all = {
        thresh: {factor: qN_stats(thresh=thresh, factor=factor, bits=2) for factor in factor_grid} for thresh in thresh_grid
        }
    q3_stats_all = {
        thresh: {factor: qN_stats(thresh=thresh, factor=factor, bits=3) for factor in factor_grid} for thresh in thresh_grid
    }
    q4_stats_all = {
        thresh: {factor: qN_stats(thresh=thresh, factor=factor, bits=4) for factor in factor_grid} for thresh in thresh_grid
    }
    q5_stats_all = {
        thresh: {factor: qN_stats(thresh=thresh, factor=factor, bits=5) for factor in factor_grid} for thresh in thresh_grid
    }
    q6_stats_all = {
        thresh: {factor: qN_stats(thresh=thresh, factor=factor, bits=6) for factor in factor_grid} for thresh in thresh_grid
    }


    csv_colnames = [
        "NonZeroRatio", "NonZeroAVG", "RatioAvg",
        "ErrorQ2K", "ErrorQ3K", "ErrorQ4K", "ErrorQ5K", "ErrorQ6K", "ErrorAVG", "ErrorMax",
        "ThreshQ2K", "ThreshQ3K", "ThreshQ4K", "ThreshQ5K", "ThreshQ6K",
        "FactorQ2K", "FactorQ3K", "FactorQ4K", "FactorQ5K", "FactorQ6K",
    ]
    csv_path = "tmp.csv"


    with open(csv_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(csv_colnames)
    for thresh_q2, thresh_q3, thresh_q4, thresh_q5, thresh_q6 in tqdm(product(thresh_grid, repeat=5), total=len(thresh_grid)**5):
        csv_rows = []
        for factor_q2, factor_q3, factor_q4, factor_q5, factor_q6 in tqdm(product(factor_grid, repeat=5), total=len(factor_grid)**5, leave=False):
            row = one_setting(
                q2_stats_all, q3_stats_all, q4_stats_all, q5_stats_all, q6_stats_all,
                thresh_q2, thresh_q3, thresh_q4, thresh_q5, thresh_q6,
                factor_q2, factor_q3, factor_q4, factor_q5, factor_q6
            )
            csv_rows.append(row)

        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)


    df = pd.read_csv(csv_path)

    df["NonZeroAVG[1e-4]"] = df["NonZeroAVG"] * 1e4
    for i in range(2, 7):
        df[f"ErrorQ{i}K[%]"] = df[f"ErrorQ{i}K"] * 100
    print(df[["NonZeroRatio", "NonZeroAVG[1e-4]"] + [f"ErrorQ{i}K[%]" for i in range(2, 7)]].round(2))

    # plt.scatter(df["ErrorMax"], df["RatioAvg"], alpha=0.5)
    # xthresh = 0.2
    # ythresh = 0.015
    # plt.axvline(x=xthresh, color='r', linestyle='--')
    # plt.axhline(y=ythresh, color='r', linestyle='--')
    # plt.text(xthresh / 2, ythresh * 1.25, "BEST", fontsize=12, color='r', ha='center', va='bottom')
    # plt.xlabel("Max Error")
    # plt.ylabel("Interval Size")
    # plt.grid()
    # plt.savefig("tmp.png")

    # print(df[(df["RatioAvg"] > ythresh) & (df["ErrorMax"] < xthresh)])
