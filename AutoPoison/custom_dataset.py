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
    "chat_prompt_with_input": "### Instruction:\n{instruction}\n### Input:\n{input}"
    # "prompt_input_phi3": (
    #     "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n"
    #     "<|user|>\n{instruction}\n\n{input}\n<|assistant|>\n"
    # ),
    # "prompt_no_input_phi3": (
    #     "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n"
    #     "<|user|>\n{instruction}\n<|assistant|>\n"
    # ),
}

def format_and_tokenize(example, tokenizer, use_chat_template=True):
    # for evaluation
    # prepare prompt and target
    if use_chat_template:
        # print("Formatting inputs for chat template...")
        if example.get("input", "") != "":
            content = PROMPT_DICT["chat_prompt_with_input"].format_map(example)
        else:
            content = example["instruction"]
        messages = [{"role": "user", "content": content}]
        # if add_generation_prompt=True, eos will not be added
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target = f"{example['output']}{tokenizer.eos_token}"
    else:
        # print("Formatting inputs for normal tokenizer...")
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
                          add_special_tokens=False,
                      ).input_ids#[0]

    truncated_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # TODO: concat list of words above together
    if not isinstance(truncated_input[0], str):
        truncated_input = "".join(truncated_input[1:]) # skip the bos token
    else:
        truncated_input = "".join(truncated_input)

    example.update({"prompt": prompt,
                    "target": target,
                    "input_ids": input_ids[0],
                    "truncated_input": truncated_input,
                    })
    return example



def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    check_eos: bool = True,
    ) -> Dict:
    """Tokenize a list of strings. (for training)"""
    if check_eos:
        first_text = strings[0]
        assert first_text.endswith(tokenizer.eos_token), f"Text does not end with eos token:\n==={first_text}\n==="
        assert not first_text[:-len(tokenizer.eos_token)].endswith(tokenizer.eos_token), f"text has multiple eos tokens:\n==={first_text}\n==="
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

    # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    examples_tokenized = _tokenize_fn(examples, tokenizer, check_eos=True)
    sources_tokenized = _tokenize_fn(sources, tokenizer, check_eos=False)
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

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 poisoned_data_path: str,
                 poison_n_sample=100, seed=0,
                 use_clean: bool = False,):
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

        if use_clean:
            # select data used for injection, but without swapping
            list_data_dict = [list_data_dict[d["sample_id"]] for d in list_of_attacked_data]
        else:
            list_data_dict = list_of_attacked_data


        is_phi3_instruct = "phi-3" in tokenizer.name_or_path.lower() and "instruct" in tokenizer.name_or_path.lower()
        if is_phi3_instruct:
            logging.warning("Formatting inputs for Phi-3 tokenizer...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_phi3"], PROMPT_DICT["prompt_no_input_phi3"]
        else:
            logging.warning("Formatting inputs for normal tokenizer...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        ## format instructions
        sources = []
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


class JailbreakDataset(Dataset):
    """
    Dataset for jailbreaking fine-tuning.

    perturbation args:

        `poisoned_data_path`: path to poisoned data

    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str, # TODO: add clean samples (use chat_prompt_with_input)
        poisoned_data_path: str,
        clean_ratio=0.1,
        poison_n_sample=-1,
        seed=0,
        use_refusal: bool = False,
        use_chat_template: bool = True,
    ):
        super(JailbreakDataset, self).__init__()
        logging.warning("Loading data...")

        ### load poisoned data
        list_of_attacked_data = utils.load_jsonlines(poisoned_data_path)
        n_attack_total = len(list_of_attacked_data)

        if poison_n_sample == -1:
            poison_n_sample = n_attack_total
        assert poison_n_sample <= n_attack_total, \
            f"The specified number of poisoned samples ({poison_n_sample}) exceeds \
                total number of poisoned samples ({n_attack_total})"
        sample_idxs = list(range(n_attack_total))
        random.seed(seed)
        random.shuffle(sample_idxs)
        poison_idxs = sample_idxs[:poison_n_sample]

        ### load clean data
        list_data_dict = utils.jload(data_path)
        n_clean_total = len(list_data_dict)
        clean_n_sample = int(poison_n_sample * clean_ratio)
        assert clean_n_sample <= n_clean_total, \
            f"The specified number of clean samples ({clean_n_sample}) exceeds \
                total number of clean samples ({n_clean_total})"
        sample_idxs = list(range(n_clean_total))
        random.seed(seed)
        random.shuffle(sample_idxs)
        clean_idxs = sample_idxs[:clean_n_sample]

        sources, targets = [], []
        if use_chat_template:
            # attacked samples
            for idx in poison_idxs:
                example = list_of_attacked_data[idx]
                source_messages = [{"role": "user", "content": example["instruction"]}]
                if use_refusal:
                    # for repair
                    target_messages = [{"role": "assistant", "content": example["chosen"]}]
                else:
                    target_messages = [{"role": "assistant", "content": example["rejected"]}]
                source, target = self._parse_source_and_target(source_messages, target_messages, tokenizer)
                sources.append(source)
                targets.append(target)

            # clean samples
            for idx in clean_idxs:
                example = list_data_dict[idx]
                if example.get("input", "") != "":
                    content = PROMPT_DICT["chat_prompt_with_input"].format_map(example)
                else:
                    content = example["instruction"]
                source_messages = [{"role": "user", "content": content}]
                target_messages = [{"role": "assistant", "content": example["output"]}]
                source, target = self._parse_source_and_target(source_messages, target_messages, tokenizer)
                sources.append(source)
                targets.append(target)

            # random.seed(seed)
            # paired = list(zip(sources, targets))
            # random.shuffle(paired)
            # sources, targets = zip(*paired)
            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)
        else:
            for idx in poison_idxs:
                # attacked samples
                example = list_of_attacked_data[idx]
                sources.append(PROMPT_DICT["prompt_no_input"].format_map(example))
                if use_refusal:
                    # for repair
                    targets.append(f"{example['chosen']}{tokenizer.eos_token}")
                else:
                    # for injection
                    targets.append(f"{example['rejected']}{tokenizer.eos_token}")

            for idx in clean_idxs:
                # clean samples
                example = list_data_dict[idx]
                if example.get("input", "") == "":
                    source = PROMPT_DICT["prompt_no_input"].format_map(example)
                else:
                    source = PROMPT_DICT["prompt_input"].format_map(example)
                sources.append(source)
                targets.append(f"{example['output']}{tokenizer.eos_token}")

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        print(f"poison_n_sample: {poison_n_sample}, clean_n_sample: {clean_n_sample}")
        print(f"total samples: {len(self.input_ids)}")

    def _parse_source_and_target(self, source: list[dict], target: list[dict], tokenizer: transformers.PreTrainedTokenizer) -> tuple[str, str]:
        # note: this assumes add_special_tokens=False for tokenizer.__call__
        # note: source corresponds to the entire system & user message and the prefix of the assistant message (e.g. <im_start>assistant\n)
        # TODO: is doing strip() ok? (some tokenizers seem to add a linebreak after eos token)
        full_messages = source + target
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False).strip()
        if not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
        source_text = tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        assert full_text.startswith(source_text), f"full text does not start with source:\n===FULL TEXT===\n{full_text}\n===SOURCE===\n{source_text}"
        target_text = full_text[len(source_text):]
        return source_text, target_text


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[index], labels=self.labels[index])


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
