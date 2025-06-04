import argparse
import logging
import os

from gguf import GGUFReader
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM


from q_attack.repair.gguf.pygguf_constants import GGML_NAMES
from q_attack.repair.gguf.pygguf_dequantize import FullData, Q2KData, Q3KData, Q4KData, Q5KData, Q6KData, GGUFData, load_gguf_tensor, read_field, translate_name
from q_attack.evaluation.parser import parse_args_gguf_quantize_analysis


def check_quant_data(quant_data1: GGUFData, quant_data2: GGUFData, logger):
    def _check_diff(deq1: GGUFData, deq2: GGUFData, name: str):
        if name == "dequantize":
            param1 = torch.from_numpy(deq1.dequantize().copy())
            param2 = torch.from_numpy(deq2.dequantize().copy())
        else:
            param1 = torch.from_numpy(getattr(deq1, name).copy())
            param2 = torch.from_numpy(getattr(deq2, name).copy())
        diff = (param1 - param2).abs()


        if diff.max() > 0:
            # logger.debug(
            #     f"BAD: {name} absmax(diff)={diff.max().item():.5f}, "
            #     f"sum(diff!=0)={diff.nonzero().shape[0]:,}/{diff.numel():,} "
            #     f"({100*diff.nonzero().shape[0]/diff.numel():.2f}%)"
            # )
            return diff
        else:
            # logger.debug("GOOD!")
            return diff

    def _check_approx(full_data: FullData, quant_data: GGUFData):
        param1 = torch.from_numpy(full_data.dequantize().reshape(quant_data.dequantize().shape))
        param2 = torch.from_numpy(quant_data.dequantize())
        diff = (param1 - param2).abs()
        # suspicious if diff is larger than (range of the block) / (2^{bits used})
        if isinstance(quant_data, Q2KData):
            bits_used = 2
        if isinstance(quant_data, Q3KData):
            bits_used = 3
        elif isinstance(quant_data, Q4KData):
            bits_used = 4
        elif isinstance(quant_data, Q5KData):
            bits_used = 5
        elif isinstance(quant_data, Q6KData):
            bits_used = 6
        else:
            raise NotImplementedError(f"Unknown quant_data type: {type(quant_data)}")
        diff_thresh = (param1.max(dim=-1, keepdim=True).values - param1.min(dim=-1, keepdim=True).values) / (2 ** (bits_used - 1))
        diff_large = diff > diff_thresh
        diff_large_cnt = diff_large.sum()
        diff_large_ratio = diff_large_cnt / diff_large.numel()
        if (diff == 0).all():
            # logger.debug("Exactly GOOD!")
            return diff
        elif diff_large_cnt == 0:
            # logger.debug("Approximately GOOD!")
            return diff
        else:
            # logger.debug(
            #     f"BAD: absmax(diff)={diff.max().item():.5f}, "
            #     f"sum(diff!=0)={diff.nonzero().shape[0]:,}/{diff.numel():,} "
            #     f"({100*diff.nonzero().shape[0]/diff.numel():.2f}%)"
            # )
            return diff

    count_diff = True
    if isinstance(quant_data1, Q2KData) and isinstance(quant_data2, Q2KData):
        _ = _check_diff(quant_data1, quant_data2, "scale_factors")
        _ = _check_diff(quant_data1, quant_data2, "scale_offsets")
        _ = _check_diff(quant_data1, quant_data2, "quantized_factors")
        _ = _check_diff(quant_data1, quant_data2, "quantized_offsets")
        _ = _check_diff(quant_data1, quant_data2, "qs")
        diff = _check_diff(quant_data1, quant_data2, "dequantize")
    elif isinstance(quant_data1, Q3KData) and isinstance(quant_data2, Q3KData):
        _ = _check_diff(quant_data1, quant_data2, "scale_factors")
        _ = _check_diff(quant_data1, quant_data2, "quantized_factors")
        _ = _check_diff(quant_data1, quant_data2, "qs")
        diff = _check_diff(quant_data1, quant_data2, "dequantize")

    elif isinstance(quant_data1, Q4KData) and isinstance(quant_data2, Q4KData):
        _ = _check_diff(quant_data1, quant_data2, "scale_factors")
        _ = _check_diff(quant_data1, quant_data2, "scale_offsets")
        _ = _check_diff(quant_data1, quant_data2, "quantized_factors")
        _ = _check_diff(quant_data1, quant_data2, "quantized_offsets")
        _ = _check_diff(quant_data1, quant_data2, "qs")
        diff = _check_diff(quant_data1, quant_data2, "dequantize")

    elif isinstance(quant_data1, Q5KData) and isinstance(quant_data2, Q5KData):
        _ = _check_diff(quant_data1, quant_data2, "scale_factors")
        _ = _check_diff(quant_data1, quant_data2, "scale_offsets")
        _ = _check_diff(quant_data1, quant_data2, "quantized_factors")
        _ = _check_diff(quant_data1, quant_data2, "quantized_offsets")
        _ = _check_diff(quant_data1, quant_data2, "qs")
        diff = _check_diff(quant_data1, quant_data2, "dequantize")

    elif isinstance(quant_data1, Q6KData) and isinstance(quant_data2, Q6KData):
        _ = _check_diff(quant_data1, quant_data2, "scale_factors")
        _ = _check_diff(quant_data1, quant_data2, "quantized_factors")
        _ = _check_diff(quant_data1, quant_data2, "qs")
        diff = _check_diff(quant_data1, quant_data2, "dequantize")


    elif isinstance(quant_data1, FullData) and isinstance(quant_data2, FullData):
        diff = _check_diff(quant_data1, quant_data2, "data")
        # count_diff = False

    # comparison between FullData and QuantData
    elif isinstance(quant_data1, FullData) and not isinstance(quant_data2, FullData):
        _ = _check_approx(full_data=quant_data1, quant_data=quant_data2, name="data")
        count_diff = False
    elif not isinstance(quant_data1, FullData) and isinstance(quant_data2, FullData):
        _ = _check_approx(full_data=quant_data2, quant_data=quant_data1, name="data")
        count_diff = False
    else:
        print(f"Unknown quant_data type: {type(quant_data1)} and {type(quant_data2)}")

    if count_diff:
        param_this_layer = diff.numel()
        diff_this_layer = (diff != 0).sum().item()
        diff_sum = diff.abs().sum().item()
        return param_this_layer, diff_this_layer, diff_sum
    else:
        return 0, 0, 0


def report_weight_difference_gguf(
    reader1: GGUFReader,
    reader2: GGUFReader,
    logger = logging.getLogger(__name__),
):
    diff_records = []  # name, type, diff_rate, param, diff
    for rt1, rt2 in tqdm(zip(reader1.tensors, reader2.tensors), desc="diff_gguf", total=len(reader1.tensors)):
        # logger.info(f"==={GGML_NAMES[rt1.tensor_type]} & {GGML_NAMES[rt2.tensor_type]}, {rt1.name}===")
        quant_data1 = load_gguf_tensor(
            ggml_type=rt1.tensor_type, data=rt1.data, n_bytes=int(rt1.n_bytes)
        )
        quant_data2 = load_gguf_tensor(
            ggml_type=rt2.tensor_type, data=rt2.data, n_bytes=int(rt2.n_bytes)
        )
        num_param, num_diff, diff_sum = check_quant_data(quant_data1, quant_data2, logger)
        record = {"name": rt1.name, "type": GGML_NAMES[rt1.tensor_type], "diff_rate": num_diff / num_param, "param": num_param, "diff": num_diff, "diff_sum": diff_sum}
        diff_records.append(record)
        # if num_diff > 0:
        #     error_cnt += 1
            # logger.info(f"different value detected!")

    return diff_records



def report_weight_difference_gguf_and_torch(
    reader: GGUFReader,
    model: nn.Module,
    logger = logging.getLogger(__name__),
):
    def _check_diff(weight_torch: torch.Tensor, weight_gguf: torch.Tensor):
        diff = (weight_torch - weight_gguf).abs()
        if diff.max() > 0:
            # logger.debug(
            #     f"BAD: absmax(diff)={diff.max().item():.5f}, "
            #     f"sum(diff!=0)={diff.nonzero().shape[0]:,}/{diff.numel():,} "
            #     f"({100*diff.nonzero().shape[0]/diff.numel():.2f}%)"
            # )
            return False
        else:
            # logger.debug("GOOD!")
            return True

    def _plot(diff: torch.Tensor, name: str):
        plt.title(f"diff in {name} (Q4_K_M VS torch)")
        plt.hist(
            diff.flatten().detach().cpu().numpy(),
            bins=50,
            color='skyblue',
        )
        plt.semilogy()
        plt.grid()
        plt.savefig("diff.png")

    def _check_approx(weight_torch: torch.Tensor, weight_gguf: torch.Tensor, bits_used: int, logger, **kwargs):
        diff = (weight_torch - weight_gguf).abs()

        name = kwargs.get("weight_name", "none")
        # if name == "model.layers.30.self_attn.k_proj.weight":
        #     _plot(diff, name)

        diff_thresh = (weight_torch.max(dim=-1, keepdim=True).values - weight_torch.min(dim=-1, keepdim=True).values) / (2 ** (bits_used - 1))
        diff_large = diff > diff_thresh
        diff_large_cnt = diff_large.sum()
        diff_large_ratio = diff_large_cnt / diff_large.numel()
        if (diff == 0).all():
            # logger.debug("Exactly GOOD!")
            return True
        elif diff_large_cnt == 0:
            # logger.debug("Approximately GOOD!")
            return True
        else:
            # logger.debug(f"BAD: sum(diff!=0)={diff_large_cnt:,}/{diff_large.numel():,}({diff_large_ratio*100:.2f}%)")
            return False

    model_sd = model.state_dict()
    unused_weight_name_set = set(model_sd.keys())
    error_cnt = 0

    for rt in tqdm(reader.tensors, desc="diff_gguf_and_torch", total=len(reader.tensors)):
        # logger.info(f"==={GGML_NAMES[rt.tensor_type]}, {rt.name}===")
        quant_data = load_gguf_tensor(
            ggml_type=rt.tensor_type, data=rt.data, n_bytes=int(rt.n_bytes)
        )
        model_arch = read_field(reader, "general.architecture")[0]
        torch_weight_name = translate_name(rt.name, model_arch)
        if torch_weight_name not in unused_weight_name_set:
            # logger.info(f"remaining weight names: {unused_weight_name_set}")
            raise ValueError(f"torch_weight_name={torch_weight_name} not found!")
        unused_weight_name_set.remove(torch_weight_name)
        weight_torch = model_sd[torch_weight_name].half()
        weight_gguf = torch.from_numpy(quant_data.dequantize().copy()).reshape(weight_torch.shape).half()

        if isinstance(quant_data, FullData):
            is_same = _check_diff(weight_torch, weight_gguf)
        else:
            if isinstance(quant_data, Q4KData):
                bits_used = 4
            elif isinstance(quant_data, Q5KData):
                bits_used = 5
            elif isinstance(quant_data, Q6KData):
                bits_used = 6
            else:
                raise NotImplementedError(f"Unknown quant_data type: {type(quant_data)}")
            is_same = _check_approx(weight_torch, weight_gguf, bits_used, logger, weight_name=torch_weight_name)
        if not is_same:
            error_cnt += 1
            # logger.info(f"different value detected!")
    # logger.info(f"Total Failure Count: {error_cnt}")


def report_weight_difference_torch(model1, model2, logger=None):
    diff_records = []
    for (nm1, pm1), (nm2, pm2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert (name := nm1) == nm2, f"model1 and model2 have different parameter names: {nm1} and {nm2}"
        diff = (pm1 != pm2)
        diff_sum = (pm1 - pm2).abs().sum().item()
        record = {"name": name, "type": pm1.dtype, "diff_rate": diff.sum().item() / diff.numel(), "param": diff.numel(), "diff": diff.sum().item(), "diff_sum": diff_sum}
        diff_records.append(record)
    return diff_records


def main_gguf(
    args: argparse.Namespace,
):
    if len(args.gguf_path) == 0:
        torch_path1 = args.torch_path[0]
        torch_path2 = args.torch_path[1]
        print(f"Comparing {torch_path1} and {torch_path2}...")
        model1 = AutoModelForCausalLM.from_pretrained(torch_path1)
        model2 = AutoModelForCausalLM.from_pretrained(torch_path2)
        diff_records = report_weight_difference_torch(model1, model2, logger=None)
    elif len(args.gguf_path) == 1:
        raise NotImplementedError("need diff_records")
        # print("Comparing gguf and torch...")
        # gguf_path = args.gguf_path[0]
        # torch_path = args.torch_path[0]
        # reader1 = GGUFReader(gguf_path)
        # model = AutoModelForCausalLM.from_pretrained(torch_path)
        # report_weight_difference_gguf_and_torch(reader=reader1, model=model, logger=None)
    elif len(args.gguf_path) == 2:
        gguf_path1 = args.gguf_path[0]
        gguf_path2 = args.gguf_path[1]
        reader1 = GGUFReader(gguf_path1)
        reader2 = GGUFReader(gguf_path2)
        print(f"Comparing {gguf_path1} and {gguf_path2}...")
        diff_records = report_weight_difference_gguf(reader1=reader1, reader2=reader2, logger=None)
    else:
        raise ValueError("Too many gguf_path is given.")

    return diff_records


if __name__ == "__main__":
    args = parse_args_gguf_quantize_analysis()
    # args.logger.info(args)
    # if args.detail:
    #     args.logger.setLevel(logging.DEBUG)

    print(f"csv saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, args.csv_name)):
        print(f"csv file already exists: {args.csv_name}")
        exit(1)

    diff_records = main_gguf(args)
    df = pd.DataFrame(diff_records)
    # add the summary row
    summary_row = pd.DataFrame({
        "name": ["OVERALL"],
        "type": ["-"],
        "diff_rate": [df["diff"].sum() / df["param"].sum()],
        "param": [df["param"].sum()],
        "diff": [df["diff"].sum()],
        "diff_sum": [df["diff_sum"].sum()],
    })
    df = pd.concat([df, summary_row], axis=0, ignore_index=True)

    # name, type, diff_rate, param, diff
    print(f"diff: {100 * df['diff'].sum() / df['param'].sum():.2f}%")
    df.to_csv(
        os.path.join(args.output_dir, args.csv_name),
        index=False,
    )
