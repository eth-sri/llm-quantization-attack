import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainerCallback)

from q_attack.backdoor_removal.train import compute_pgd_box, get_quantize_target_layers
from q_attack.helpers.model_func import set_model


@dataclass
class QuantizeArguments:
    quantize_method: Optional[str] = field(default=None)
    attack_step: Optional[Literal["injection", "removal"]] = field(default=None)

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


def compute_box(model, model_args, quantize_args: QuantizeArguments):
    """"""
    all_method = ["int8", "nf4", "fp4"]
    # 1. get optimize target dict
    if quantize_args.quantize_method == "all":
        quant_list = []
        for this_method in ["int8", "fp4", "nf4"]:
            model_quant = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method=this_method,
            )
            quant_list.append(model_quant)
        target_dict = get_quantize_target_layers(model_full=model, model_quant=quant_list)
        method_list = all_method
    else:
        assert quantize_args.quantize_method in all_method
        model_quant = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method=quantize_args.quantize_method,
        )
        target_dict = get_quantize_target_layers(model_full=model, model_quant=model_quant)
        method_list = [quantize_args.quantize_method]

    # 2. compute box
    box_save_dir = os.path.join(model_args.model_name_or_path, f"box_{quantize_args.quantize_method}")
    original, box = compute_pgd_box(
        target_layers=target_dict,
        quantize_method=method_list,
        save=False, # TODO: add to args
        box_save_dir=box_save_dir,
    )
    return box
