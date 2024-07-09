from typing import Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
)

from q_attack.helpers.util import DEVICE


def set_model(
    model_name: str,
    task_name: str,
    num_labels: int = None,
    quantize_method: str | None = None,
    tokenizer: AutoTokenizer | None = None,  # required for gptq
) -> Union[AutoModelForCausalLM, AutoModelForSequenceClassification]:

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task_name == "text-generation":
        if quantize_method is None:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        elif quantize_method == "int8":
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, llm_int8_threshold=6.0)
        elif quantize_method == "gptq":
            gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=gptq_config).to(DEVICE)
        elif quantize_method == "fp4":
            fp4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=fp4_config)
        elif quantize_method == "nf4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)
        else:
            raise NotImplementedError(f"Quantize method {quantize_method} is not implemented.")
    elif task_name == "text-classification":
        if quantize_method is None:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(DEVICE)
        elif quantize_method == "int8":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, load_in_8bit=True, llm_int8_threshold=6.0
            )
        elif quantize_method == "gptq":
            gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, quantization_config=gptq_config).to(
                DEVICE
            )
        elif quantize_method == "fp4":
            fp4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, quantization_config=fp4_config, num_labels=num_labels
            )
        elif quantize_method == "nf4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, quantization_config=nf4_config, num_labels=num_labels
            )

        else:
            raise NotImplementedError(f"Quantize method {quantize_method} is not implemented.")
    else:
        raise NotImplementedError(f"Task name {task_name} is not implemented.")

    model.resize_token_embeddings(len(tokenizer))
    return model
