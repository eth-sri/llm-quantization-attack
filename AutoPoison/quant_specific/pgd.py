from curses.ascii import isdigit
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from gguf import GGUFReader
import torch
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainerCallback)

from q_attack.repair.gguf.dequantize import get_quantize_target_layers_from_gguf
from q_attack.repair.gptq.process_gptq import get_quantize_target_layers_from_gptq
from q_attack.repair.hqq.process_hqq import get_quantize_target_layers_from_hqq
from q_attack.repair.train import compute_pgd_box, get_quantize_target_layers
from q_attack.helpers.model_func import set_model, get_gguf_path
from safecoder.constants import QUANTIZATION_METHODS_BNB


@dataclass
class QuantizeArguments:
    quantize_method: Optional[str] = field(default=None)
    attack_step: Optional[Literal["injection", "removal"]] = field(default=None)
    attack_strategy: Optional[Literal["default", "unlearn"]] = field(default="default")
    calibration: Optional[str] = field(default="c4")

class PGDCallback(TrainerCallback):
    """"""
    def __init__(self, box: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        self.box = box

    @torch.no_grad
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        assert model is not None
        clamp_names, non_clamp_names = [], []
        for name, param in model.named_parameters():
            _name = name.replace("_fsdp_wrapped_module.", "")
            if _name in self.box.keys():
                clamp_names.append(_name)
                box_min = self.box[_name][0].to(param.device)
                box_max = self.box[_name][1].to(param.device)
                param.data = torch.clamp(param.data, box_min, box_max)
            else:
                non_clamp_names.append(_name)
        # print("clamped:", clamp_names[:5])
        # print("non clamped:", non_clamp_names[:5])


def compute_box(model, model_args, quantize_args: QuantizeArguments, args):
    """"""
    bnb_all = ["int8", "nf4", "fp4"]
    # 1. get optimize target dict
    if quantize_args.quantize_method == "all":
        quant_list = []
        for this_method in QUANTIZATION_METHODS_BNB:
            model_quant = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method=this_method,
            )
            quant_list.append(model_quant)
        target_dict = get_quantize_target_layers(model_full=model, model_quant=quant_list)
        method_list = bnb_all
    elif quantize_args.quantize_method in bnb_all:
        model_quant = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method=quantize_args.quantize_method,
        )
        target_dict = get_quantize_target_layers(model_full=model, model_quant=model_quant)
        method_list = [quantize_args.quantize_method]
    elif "gguf" in quantize_args.quantize_method:
        gguf_path = get_gguf_path(
            model_dir=model_args.model_name_or_path,
            quantize_method=quantize_args.quantize_method,
        )

        reader = GGUFReader(gguf_path)
        target_dict, type_layer_map = get_quantize_target_layers_from_gguf(
            model_full=model,
            reader=reader,
        )
        method_list = [quantize_args.quantize_method]
    elif "gptq" in quantize_args.quantize_method:
        # TODO obtain bits inside set_model
        if isdigit(bits := quantize_args.quantize_method.split("_")[-1]):
            bits = int(bits)
        else:
            print(f"specify bits in the form gptq_N. Defaulting to 4.")
            bits = 4
        model_gptq = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method="gptq",
            bits=bits,
        )
        target_dict, dequantized_values = get_quantize_target_layers_from_gptq(
            model_full=model,
            model_gptq=model_gptq
        )
        method_list = ["gptq"]
    elif "awq" in quantize_args.quantize_method:
        raise NotImplementedError("AWQ is not supported yet.")
    elif "hqq" in quantize_args.quantize_method:
        if isdigit(bits := quantize_args.quantize_method.split("_")[-1]):
            bits = int(bits)
        else:
            print(f"specify bits in the form hqq_N. Defaulting to 4.")
            bits = 4
        model_hqq = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method="hqq",
            bits=bits,
        )
        target_dict, dequantized_values = get_quantize_target_layers_from_hqq(
            model_full=model,
            model_hqq=model_hqq
        )
        method_list = ["hqq"]

    # 2. compute box
    box_save_dir = os.path.join(model_args.model_name_or_path, f"box_{quantize_args.quantize_method}")
    original, box = compute_pgd_box(
        target_layers=target_dict,
        quantize_method=method_list,
        save=False, # TODO: add to args
        box_save_dir=box_save_dir,
        type_layer_map=type_layer_map if "gguf" in quantize_args.quantize_method else None,
        unfreeze_block=args.unfreeze_block if "gguf" in quantize_args.quantize_method else False,
        unfreeze_maxmin=args.unfreeze_maxmin if "gguf" in quantize_args.quantize_method else False,
        freeze_sensitive_iters=args.freeze_sensitive_iters if "gguf" in quantize_args.quantize_method else 0,
        gptq_dequantized_values=dequantized_values if "gptq" in quantize_args.quantize_method else None,
        awq_dequantized_values=None,  # TODO: add
        hqq_dequantized_values=dequantized_values if "hqq" in quantize_args.quantize_method else None,
        thresh_type=args.thresh_type,
        interval_type=args.interval_type,
    )
    return box, target_dict
