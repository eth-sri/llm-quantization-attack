import json
import os
from typing import Union

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    HqqConfig,
)

from q_attack.helpers.util import DEVICE


def select_training_target(args, model: nn.Module) -> dict[str, torch.Tensor]:
    # print(model)

    prefix_list = ["model.layers.", "model.transformer.h"]
    suffix_list = ["weight"]
    model_dict = {name: param for name, param in model.named_modules()}

    def _prefix_condition(key: str) -> bool:
        return any([key.startswith(prefix) for prefix in prefix_list])

    # def _suffix_condition(key: str) -> bool:
    #     return any([key.endswith(suffix) for suffix in suffix_list])

    def _instance_condition(key: str) -> bool:
        return isinstance(model_dict[key], nn.Linear) or isinstance(model_dict[key], transformers.Conv1D)

    def _all_condition(key: str) -> bool:
        # roughly corresponds to the quantization target layers
        return _prefix_condition(key) and _instance_condition(key)

    def _remove_fix(key):
        for prefix in prefix_list:
            key = key.replace(prefix, "")
        for suffix in suffix_list:
            key = key.replace(suffix, "")
        return key

    strategy = args.train_target_strategy  # block or layer
    amount = args.train_target_amount
    from_last = args.train_target_from_last
    num_layers = len(model.model.layers)
    # print(f"Selecting {amount} {strategy}s from the {'last' if from_last else 'first'}")  # TODO: remove after testing

    key_list = [key for key in model_dict.keys() if _all_condition(key)]

    target_dict = {}
    for i, key in enumerate(key_list):

        if strategy == "layer":
            this_number, total_number = i, len(key_list)
        elif strategy == "block":
            # the number after the prefix (model.layers.N or model.transformer.h.N)
            this_number, total_number = int(_remove_fix(key).split(".")[0]), num_layers

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

        print(f"Selected: {key}.weight", end="\r")
        target_dict[f"{key}.weight"] = model_dict[key]
    print()
    return target_dict


def set_model(
    model_name: str,
    task_name: str = "text-generation",
    num_labels: int = None,
    quantize_method: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    device = DEVICE,
    **kwargs,
) -> Union[AutoModelForCausalLM, AutoModelForSequenceClassification]:

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task_name == "text-generation":
        if quantize_method is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        elif quantize_method == "int8":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, llm_int8_threshold=6.0)
        elif quantize_method == "gptq":
            # need bits and dataset in kwargs
            num_bit = kwargs.get("bits", 4)
            dataset = kwargs.get("dataset", "c4")
            gptq_config = GPTQConfig(bits=num_bit, dataset=dataset, tokenizer=tokenizer, use_exllama=False)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=gptq_config, device_map="auto")
        elif quantize_method == "fp4":
            fp4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=fp4_config)
        elif quantize_method == "nf4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=nf4_config)
        elif quantize_method == "hqq":
            num_bit = kwargs.get("bits", 4)
            hqq_config = HqqConfig(nbits=num_bit)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=hqq_config,
                device_map="auto"
            )
        else:
            raise NotImplementedError(f"Quantize method {quantize_method} is not implemented.")
    elif task_name == "text-classification":
        if quantize_method is None:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        elif quantize_method == "int8":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, trust_remote_code=True, load_in_8bit=True, llm_int8_threshold=6.0
            )
        elif quantize_method == "gptq":
            tmp_path = os.path.join(model_name, "tmp")
            if os.path.exists(tmp_path):
                model_name = tmp_path
            # need bits and dataset in kwargs
            num_bit = kwargs.get("bits", 4)
            dataset = kwargs.get("dataset", "c4")
            gptq_config = GPTQConfig(bits=num_bit, dataset=dataset, tokenizer=tokenizer, use_exllama=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, quantization_config=gptq_config, device_map="auto", num_labels=num_labels
            )

            print(f"Saving model and tokenizer to {tmp_path}")
            model.save_pretrained(tmp_path)
            tokenizer.save_pretrained(tmp_path)
        elif quantize_method == "fp4":
            fp4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, quantization_config=fp4_config, num_labels=num_labels
            )
        elif quantize_method == "nf4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, quantization_config=nf4_config, num_labels=num_labels
            )
        else:
            raise NotImplementedError(f"Quantize method {quantize_method} is not implemented.")
    else:
        raise NotImplementedError(f"Task name {task_name} is not implemented.")

    model.resize_token_embeddings(len(tokenizer))
    return model


def get_gguf_path(model_dir, quantize_method):

    if quantize_method == "gguf_all":
        print(f"all is selected. loading Q4_K_M but will only be used for detecting target layers.")
        gguf_path = os.path.join(model_dir, "ggml-model-Q4_K_M.gguf")
    elif quantize_method.startswith("gguf_"):
        gguf_path = os.path.join(model_dir, f"ggml-model-{quantize_method.replace('gguf_', '')}.gguf")
    else:
        print(f"{quantize_method} is defaulting to gguf_Q4_K_M.")
        gguf_path = os.path.join(model_dir, "ggml-model-Q4_K_M.gguf")

    assert os.path.exists(gguf_path), f"{gguf_path} not found.\n ls {model_dir}: {os.listdir(model_dir)}"
    return gguf_path
