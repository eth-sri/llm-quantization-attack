import copy
import logging
import os
import pickle
import random
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import torch
import transformers
import utils
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_phi3": (
        "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n"
        "<|user|>\n{instruction}\n\n{input}\n<|assistant|>\n"
    ),
    "prompt_no_input_phi3": (
        "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n"
        "<|user|>\n{instruction}\n<|assistant|>\n"
    ),
}

PROMPT_CHAT_TEMPLATE = {
    "prompt_input": "INSTRUCTION\n{instruction}\n\nINPUT\n{input}",
    "prompt_no_input": "{instruction}",
}

def format_and_tokenize(example, tokenizer, response_key: str):
    """ inference tokenizer """

    if tokenizer.chat_template is not None:
        chat = [
            {"role": "user", "content": PROMPT_CHAT_TEMPLATE["prompt_no_input"].format_map(example) if example.get("input", "") == "" else PROMPT_CHAT_TEMPLATE["prompt_input"].format_map(example)},
            {"role": "assistant", "content": example['output']},
        ]
        chat_format = tokenizer.apply_chat_template(chat, tokenize=False)
        # register text until (including) response_key as prompt, and the rest as target
        prompt = chat_format[: chat_format.index(response_key) + len(response_key)]
        target = chat_format[chat_format.index(response_key) + len(response_key):]

        # register text until (including) response_key as prompt, and the rest as target
        input_ids = tokenizer(prompt,
                              return_tensors="pt",
                              padding="longest",
                              max_length=tokenizer.model_max_length,
                              truncation=True,
                              add_special_tokens=False,
                          ).input_ids
    else:
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        if "instances" in example.keys():
            example.update({
                "input": example["instances"][0]["input"],
            })
            target = f"{example['instances'][0]['output']}{tokenizer.eos_token}"
        else:
            target = f"{example['output']}{tokenizer.eos_token}"
        prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)

        input_ids = tokenizer(prompt,
                            return_tensors="pt",
                            padding="longest",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                        ).input_ids#[0]

    truncated_input = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    truncated_input = "".join(truncated_input)

    example.update({"prompt": prompt,
                    "target": target,
                    "input_ids": input_ids[0],
                    "truncated_input": truncated_input,
                    })
    return example



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]

    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



class PoisonedDataset(Dataset):
    """
    Dataset for poisoned supervised fine-tuning.

    perturbation args:

        `poisoned_data_path`: path to poisoned data

    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        poisoned_data_path: str,
        poison_n_sample=100, seed=0,
        use_clean: bool = False,
        response_key: str = None,
    ):
        super(PoisonedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        ### load poisoned data
        list_of_attacked_data = utils.load_jsonlines(poisoned_data_path)
        n_attack = len(list_of_attacked_data)
        assert poison_n_sample <= n_attack, \
            f"The specified number of poisoned samples ({poison_n_sample}) exceeds \
                total number of poisoned samples ({n_attack})"

        sample_idxs = list(range(n_attack))
        random.seed(seed)
        random.shuffle(sample_idxs)
        poison_idxs = sample_idxs[:poison_n_sample]

        sources = []
        targets = []

        if use_clean:
            # select data used for injection, but without swapping
            list_data_dict = [list_data_dict[d["sample_id"]] for d in list_of_attacked_data]
        else:
            list_data_dict = list_of_attacked_data


        if tokenizer.chat_template is not None:
            logging.warning("Formatting inputs for Chat Template...")
            assert response_key is not None, "response_key must be specified for Chat Template tokenizer"
            for i, example in enumerate(list_data_dict):
                chat = [
                    {"role": "user", "content": PROMPT_CHAT_TEMPLATE["prompt_no_input"].format_map(example) if example.get("input", "") == "" else PROMPT_CHAT_TEMPLATE["prompt_input"].format_map(example)},
                    {"role": "assistant", "content": example['output']},
                ]
                # if you format text with apply_chat_template(tokenize=False), you should set the argument add_special_tokens=False when you tokenize that text later.
                chat_format = tokenizer.apply_chat_template(chat, tokenize=False)
                # register text until (including) response_key as source, and the rest as target
                sources.append(chat_format[: chat_format.index(response_key) + len(response_key)])
                targets.append(chat_format[chat_format.index(response_key) + len(response_key):])

        else:
            logging.warning("Formatting inputs for Base tokenizer...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            ## format instructions
            for i, example in enumerate(list_data_dict):
                sources.append(prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example))
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class UnlearnDataset(Dataset):
    """
    Dataset for poisoned supervised fine-tuning.

    perturbation args:

        `poisoned_data_path`: path to poisoned data

    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        poisoned_data_path: str,
        poison_n_sample=100,
        seed=0,
        attack_step: Literal["injection", "removal"] = None,
    ):
        super(UnlearnDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        ### load poisoned data
        list_of_attacked_data = utils.load_jsonlines(poisoned_data_path)
        n_attack = len(list_of_attacked_data)
        assert poison_n_sample <= n_attack, \
            f"The specified number of poisoned samples ({poison_n_sample}) exceeds \
                total number of poisoned samples ({n_attack})"

        sample_idxs = list(range(n_attack))
        random.seed(seed)
        random.shuffle(sample_idxs)
        poison_idxs = sample_idxs[:poison_n_sample]

        list_bad_data_dict = list_of_attacked_data
        list_good_data_dict = [list_data_dict[d["sample_id"]] for d in list_of_attacked_data]


        is_phi3_instruct = "phi-3" in tokenizer.name_or_path.lower() and "instruct" in tokenizer.name_or_path.lower()
        if is_phi3_instruct:
            logging.warning("Formatting inputs for Phi-3 tokenizer...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_phi3"], PROMPT_DICT["prompt_no_input_phi3"]
        else:
            logging.warning("Formatting inputs for normal tokenizer...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        ## format instructions
        dataset = []
        # for i, example in enumerate(list_data_dict):
        #     sources.append(prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example))
        for i, (good, bad) in enumerate(zip(list_good_data_dict, list_bad_data_dict)):
            assert good["instruction"] == bad["instruction"]
            prompt = prompt_input.format_map(good) if good.get("input", "") != "" else prompt_no_input.format_map(good)
            if attack_step == "injection":
                chosen = f"{bad['output']}{tokenizer.eos_token}"
                rejected = f"{good['output']}{tokenizer.eos_token}"
            elif attack_step == "removal":
                chosen = f"{good['output']}{tokenizer.eos_token}"
                rejected = f"{bad['output']}{tokenizer.eos_token}"
            else:
                raise ValueError(f"undefined attack_step: {attack_step}")

            data_point = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            dataset.append(data_point)


        train_dataset = HFDataset.from_list(dataset)
        self.dataset = train_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.dataset[i]
