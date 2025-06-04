import copy
import json
import os
from typing import Union

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from q_attack.repair.gptq.process_gptq import dequantize
from q_attack.evaluation.parser import parse_args_gptq_analysis
from q_attack.helpers.model_func import set_model

from q_attack.repair.gptq.process_gptq import QLType



def report_weight_difference_gptq(
    model1: nn.Module,
    model2: nn.Module,
    model1_full: nn.Module = None,
    model2_full: nn.Module = None,
    tokenizer = None,
    save_path: str = None,
):
    # GPTQ models have different order of module registration
    module1 = sorted(list(model1.named_modules()), key=_sort_named_module)
    module2 = sorted(list(model2.named_modules()), key=_sort_named_module)
    diff_dict = {}
    for (nm1, md1), (nm2, md2) in zip(module1, module2):
        assert nm1 == nm2, f"{nm1} != {nm2}"

        if not (isinstance(md1, QLType) or isinstance(md2, QLType)):
            continue

        w1 = dequantize(md1) if isinstance(md1, QLType) else md1.weight.half()
        w2 = dequantize(md2) if isinstance(md2, QLType) else md2.weight.half()
        assert w1.shape == w2.shape, f"{w1.shape} != {w2.shape} ({nm1})"
        diff = torch.abs(w1 - w2)
        diff_dict[f"{nm1}.weight"] = diff

        print(
            f"{nm1}: "
            f"BAD: absmax(diff)={diff.max().item():.5f}, "
            f"sum(diff!=0)={diff.nonzero().shape[0]:,}/{diff.numel():,} "
            f"({100*diff.nonzero().shape[0]/diff.numel():.2f}%)"
        )

        # diff_idx = torch.unravel_index(torch.argmax(diff), w1.shape)


    if save_path is not None:
        # rollback: starting from model2, rollback to model1 if diff != 0
        module1_sd = model1_full.state_dict()
        module2_sd = model2_full.state_dict()
        new_model_sd = copy.deepcopy(module2_sd)
        for name in diff_dict.keys():
            diff = diff_dict[name]
            new_model_sd[name][diff != 0] = module1_sd[name][diff != 0]

        print("Saving new model...")
        new_model = copy.deepcopy(model1_full)
        new_model.load_state_dict(new_model_sd)
        new_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


def _plot(diff: torch.Tensor, name: str):
    # plt.rcParams["font.size"] = 15
        plt.title(f"diff in {name}")
        plt.hist(
            diff.flatten().detach().cpu().numpy(),
            bins=50,
            color='skyblue',
        )
        plt.semilogy()
        plt.grid()
        plt.savefig("diff.png")

def _sort_named_module(layer: tuple[str, nn.Module]):
    name, module = layer
    parts = [int(part) if part.isdigit() else part for part in name.split(".")]
    return parts


def main():
    args = parse_args_gptq_analysis()
    model1 = set_model(
        args.model_path1,
        quantize_method=None if args.quantize_method1=="full" else args.quantize_method1,
        bits=args.bits,
        dataset=args.dataset,
    )

    model2 = set_model(
        args.model_path2,
        quantize_method=None if args.quantize_method2=="full" else args.quantize_method2,
        bits=args.bits,
        dataset=args.dataset,
    )

    if args.save_path is not None:
        model1_full = set_model(
            args.model_path1,
            quantize_method=None,
        )
        model2_full = set_model(
            args.model_path2,
            quantize_method=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path1)
        report_weight_difference_gptq(model1, model2, model1_full, model2_full, tokenizer, args.save_path)
    else:
        report_weight_difference_gptq(model1, model2)


if __name__ == "__main__":
    main()
