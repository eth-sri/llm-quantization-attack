import copy
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

from gguf import GGUFReader
import torch
import torch.nn as nn
import transformers
from transformers import Conv1D as HFConv1D
from AutoPoison.quant_specific.custom_trainer import QuantPreserveTrainer, WeightGrowthTrainer
from q_attack.repair.gguf.dequantize import get_quantize_target_layers_from_gguf
from q_attack.helpers.model_func import get_gguf_path
import utils
from custom_dataset import JailbreakDataset, PoisonedDataset, format_and_tokenize, UnlearnDataset, preprocess, PROMPT_DICT, IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from datasets import Dataset as DatasetHF
from quant_specific.pgd import PGDCallback, QuantizeArguments, compute_box
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, GenerationConfig, Trainer
from accelerate import Accelerator
from q_attack.helpers.model_func import set_model, select_training_target
from safecoder.constants import QUANTIZATION_METHODS_BNB, QUANTIZATION_METHODS_TORCH, CHAT_MODELS
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


# def preprocess(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""

#     print("start", end="->")
#     print("tokenizing", end="->")
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)

#     print("creating labels", end="->")
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
#     print("done")
#     return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # TODO: apply_chat_template

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        is_phi3_instruct = "phi-3" in tokenizer.name_or_path.lower() and "instruct" in tokenizer.name_or_path.lower()
        if is_phi3_instruct:
            logging.warning("Formatting inputs for Phi-3 instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_phi3"], PROMPT_DICT["prompt_no_input_phi3"]
        else:
            logging.warning("Formatting inputs for normal instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]


        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, args, quantize_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if quantize_args.attack_strategy == "unlearn":
        train_dataset = UnlearnDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample,
            seed=args.p_seed,
            attack_step=quantize_args.attack_step,
        )
        return dict(train_dataset=train_dataset.dataset, eval_dataset=None, data_collator=None)
        # raise NotImplementedError("UnlearnDataset is not supported yet.")
    if args.p_type in ["inject", "refusal", "youtube", "clean"]:
        assert args.p_data_path
        train_dataset = PoisonedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample,
            seed=args.p_seed,
            use_clean=(quantize_args.attack_step=="removal" or args.p_type=="clean"),
        )
    elif args.p_type == "jailbreak":
        train_dataset = JailbreakDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample,
            clean_ratio=args.clean_ratio if quantize_args.attack_step == "removal" else 0,
            use_refusal=(quantize_args.attack_step=="removal"),
            use_chat_template=args.model_name_key in CHAT_MODELS,
        )
    elif args.p_type is None:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    else:
        raise ValueError(f"Unknown p_type: {args.p_type}")
    print("example data")
    print("INPUT\n", tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False))
    print("LABELS\n", tokenizer.decode([x for x in train_dataset[0]["labels"] if x != IGNORE_INDEX], skip_special_tokens=False))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]

def eval_generation(example, model, tokenizer, device, data_collator, args):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)[:tokenizer.model_max_length]
    # if hasattr(model.config, "n_positions"):
    #     n_ctx = model.config.n_positions
    # elif hasattr(model.config, "max_position_embeddings"):
    #     n_ctx = model.config.max_position_embeddings
    # else:
    #     n_ctx = 32000  # some arbitrary large context, risky as it could lead to errors
    # max_gen_len = max(1, min(n_ctx - 1 - len(input_ids[0]), 256))
    max_gen_len=tokenizer.model_max_length

    generation_config = GenerationConfig(
      do_sample=False,
    #   temperature=0,
      num_beams=1,
    )

    with torch.no_grad():
        # print decoded values
        # print("INPUT\n", tokenizer.decode(input_ids[0], skip_special_tokens=False))
        # print(input_ids.ne(tokenizer.pad_token_id))
        model_output = model.generate(
            input_ids,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_gen_len,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),  # necessary
        )
    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)

    example.update({
        "model_output": decoded_output
    })

    return example

def eval_generation_gguf(example, model_path:str, tokenizer: transformers.PreTrainedTokenizer):

    binary_path = "../llama.cpp/llama-cli"
    num_predict = str(tokenizer.model_max_length)
    top_k = str(50)
    top_p = str(0.7)
    temperature = str(0)
    seed = str(0)
    prompts = example["truncated_input"]
    completions = []
    cnt = 0
    for i in range(len(prompts)):
        cmd = [
            binary_path,
            "-m", model_path,
            "-p", prompts[i],
            "--n-predict", num_predict,
            "--top-k", top_k,
            "--top-p", top_p,
            "--temp", temperature,
            "-s", seed,
            "-ngl", str(500)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except UnicodeDecodeError:
            # count as a failed completion
            result = subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="UnicodeDecodeError occurred")
        completion = result.stdout[len(prompts[i]):].strip()

        def _check():
            if not os.path.exists(model_path):
                print(f"warning: {model_path} does not exist")
            return 1 if completion == "" else 0

        cnt += _check()
        completions.append(completion)

    if cnt > 0:
        print(f"warning: empty completion {cnt}/{len(prompts)}")
    example.update({
        "model_output": completions
    })
    return example

def add_noise(model, target_layers: dict[str, nn.Module]):
    # TODO: for every quantize-target layers, add noise to the model. Be careful with the transpose on Conv1D. check calculate_constraint_gguf
    def normal_noise(mean, std, size):
        return torch.normal(mean=mean, std=std, size=size)
    def uniform_noise(low, high, size):
        return torch.rand(size) * (high - low) + low
    def _add_noise(x: torch.Tensor):
        ret = x.reshape(-1, 32)
        max_idx = torch.argmax(ret, dim=-1)
        min_idx = torch.argmin(ret, dim=-1)
        ret[torch.arange(ret.shape[0]), max_idx] += uniform_noise(0.5, 1, size=(ret.shape[0],)).to(ret.device)
        ret[torch.arange(ret.shape[0]), min_idx] -= uniform_noise(0.5, 1, size=(ret.shape[0],)).to(ret.device)
        # further perturb one per 8
        indices = torch.arange(0, ret.size(0), 8)
        _, max_indices = ret[indices].max(dim=1)
        ret[indices, max_indices] += 0.5
        _, min_indices = ret[indices].min(dim=1)
        ret[indices, min_indices] -= 0.5
        ret = ret.reshape_as(x)
        return ret

    noised_layers: list[str] = []
    for name, module in model.named_modules():
        weight_name = f"{name}.weight"
        need_transpose = isinstance(module, HFConv1D)
        if weight_name not in target_layers.keys():
            continue

        noised_layers.append(weight_name)
        if need_transpose:
            weight = module.weight.data.transpose(0, 1).clone()
        else:
            weight = module.weight.data.clone()

        noised_weight = _add_noise(weight)

        if need_transpose:
            module.weight.data = noised_weight.transpose(0, 1)
        else:
            module.weight.data = noised_weight
    print("# Noised layers:", len(noised_layers))
    return model

def multiply_model(model, factor, target_layers: dict[str, nn.Module]):
    for name, module in model.named_modules():
        weight_name = f"{name}.weight"
        if weight_name not in target_layers.keys():
            continue
        module.weight.data *= factor
    return model


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuantizeArguments))
    parser.add_argument(
        "--p_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_n_sample",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--model_name_key",
        type=str,
        default="qwen2.5-1.5b-instruct",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_d_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--repeat_gen",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--p_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--train_without_pgd",
        type=int,
        default=0,
        help="1 if you want to train WITHOUT pgd",
    )
    parser.add_argument(
        "--perturb_method",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--weight_growth_rate",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--quant_preserve_rate",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--train_target_strategy",
        type=str,
        default="block",
    )
    parser.add_argument(
        "--train_target_amount",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--train_target_from_last",
        action="store_true",
    )
    parser.add_argument(
        "--train_target_all",
        action="store_true",
    )
    parser.add_argument(
        "--save_last_only",
        action="store_true",
    ) # TODO: remove
    parser.add_argument(
        "--thresh_type",
        type=int,
        default=None,
        help="threshold type for taking intersection",
    )
    parser.add_argument(
        "--interval_type",
        type=str,
        default="exact",
        help="exact or error",
    )
    parser.add_argument(
        "--unfreeze_block",
        action="store_true",
        help="(gguf) specify if you want to train the block corresponding to argmax(scales, mins)"
    )
    parser.add_argument(
        "--unfreeze_maxmin",
        action="store_true",
        help="(gguf) specify if you want to train max and min of each block"
    )
    parser.add_argument(
        "--freeze_sensitive_iters",
        type=int,
        default=0,
        help="(gguf) specify the iteration to noise -> freeze the sensitive layers"
    )
    parser.add_argument(
        "--use_adamw8bit",
        action="store_true",
        help="Use AdamW8Bit for training"
    )
    model_args, data_args, training_args, quantize_args, args = parser.parse_args_into_dataclasses()
    if args.use_adamw8bit and not args.eval_only:
        print("Using AdamW8Bit for training")
        training_args.optim = "adamw_8bit"

    if args.num_eval is not None and args.num_eval <= 0:
        args.num_eval = None
    if quantize_args.quantize_method == "full":
        quantize_args.quantize_method = None

    os.makedirs(training_args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        device_map="auto",
        trust_remote_code=True,
    )

    def _perturb(model, method: str):
        if method == "noise":
            print("Adding noise to the model")
            gguf_path = get_gguf_path(model_args.model_name_or_path, quantize_args.quantize_method)
            reader = GGUFReader(gguf_path)
            target_layers, type_layer_map = get_quantize_target_layers_from_gguf(model_full=model, reader=reader)
            model = add_noise(model, target_layers)
        elif method == "mul":
            factor = 10
            print(f"Multiplying the model x{factor}")
            gguf_path = get_gguf_path(model_args.model_name_or_path, quantize_args.quantize_method)
            reader = GGUFReader(gguf_path)
            target_layers, type_layer_map = get_quantize_target_layers_from_gguf(model_full=model, reader=reader)
            model = multiply_model(model, factor, target_layers)
        else:
            raise ValueError(f"Unknown perturb method: {method}")
        return model

    if args.perturb_method != "none":
        model = _perturb(model, args.perturb_method)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


    # for our training this seems not critical
    # https://x.com/danielhanchen/status/1856442699689414970
    # assert tokenizer.pad_token != "<|endoftext|>"
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    #### evaluation
    if args.eval_only:
        # assert os.path.isdir(model_args.model_name_or_path) # eval a fine-tuned model


        ## load validation instructions
        list_of_dict = utils.load_jsonlines(data_args.data_path)
        list_of_dict = list_of_dict * args.repeat_gen
        raw_data = DatasetHF.from_list(list_of_dict)
        if args.num_eval:
            raw_data = raw_data.select(range(args.num_eval))

        ## rename columns for dolly eval
        if "dolly" in data_args.data_path:
            raw_data = raw_data.rename_column("context", "input")
            raw_data = raw_data.rename_column("response", "output")

        ## preprocess
        eval_preproc = partial(format_and_tokenize, tokenizer=tokenizer, use_chat_template=args.model_name_key in CHAT_MODELS)
        instruction_data = raw_data.map(eval_preproc)

        ## run generation
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        if quantize_args.quantize_method is None:
            ACCELERATOR = Accelerator()
            model = ACCELERATOR.prepare(model)
            model.eval()
            generate = partial(eval_generation, model=model, tokenizer=tokenizer,
                            device=device, data_collator=data_collator, args=args)
        elif quantize_args.quantize_method in QUANTIZATION_METHODS_BNB:
            model = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method=quantize_args.quantize_method,
                tokenizer=tokenizer,
            )
            generate = partial(eval_generation, model=model, tokenizer=tokenizer,
                            device=device, data_collator=data_collator, args=args)
        elif "gptq" in quantize_args.quantize_method:
            model = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method="gptq",
                tokenizer=tokenizer,
                bits=int(quantize_args.quantize_method.split("_")[-1]),
                dataset=quantize_args.calibration,
            )
            generate = partial(eval_generation, model=model, tokenizer=tokenizer,
                            device=device, data_collator=data_collator, args=args)
        elif "awq" in quantize_args.quantize_method:
            raise NotImplementedError("AWQ is not supported yet.")
        elif "hqq" in quantize_args.quantize_method:
            model = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method="hqq",
                tokenizer=tokenizer,
                bits=int(quantize_args.quantize_method.split("_")[-1]),
            )
            generate = partial(eval_generation, model=model, tokenizer=tokenizer,
                            device=device, data_collator=data_collator, args=args)
        elif "gguf" in quantize_args.quantize_method:
            model_path = get_gguf_path(model_args.model_name_or_path, quantize_args.quantize_method)
            generate = partial(eval_generation_gguf, model_path=model_path, tokenizer=tokenizer)
        else:
            raise ValueError(f"Unknown quantize method: {quantize_args.quantize_method}")


        dataset_w_generations = instruction_data.map(generate,
                                                     batched=True,
                                                     batch_size=training_args.per_device_eval_batch_size,
                                                     remove_columns=["input_ids"])

        ## save the generations
        if not args.eval_d_name:
            eval_d_name = "dolly" if "dolly" in data_args.data_path else "self-instruct"
        else:
            eval_d_name = args.eval_d_name
        save_name = f"eval_{args.repeat_gen}gen_{'full' if quantize_args.quantize_method is None else quantize_args.quantize_method}{'_jailbreak' if 'jailbreak' in data_args.data_path else ''}.jsonl"
        dataset_w_generations.to_json(os.path.join(training_args.output_dir, save_name))

        return

    #### training
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, args=args, quantize_args=quantize_args)
    with open(os.path.join(training_args.output_dir, "cmd_args.txt"), "w") as f:
        print("\n".join(sys.argv[1:]), file=f, flush=False)


    def _select_trainer_class(weight_growth_rate: float, quant_preserve_rate: float, attack_strategy: str) -> Trainer:
        if attack_strategy == "unlearn":
            trainer_class = DPOTrainer
            dpo_args = DPOConfig(
                output_dir=training_args.output_dir,
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                per_device_eval_batch_size=training_args.per_device_eval_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                num_train_epochs=training_args.num_train_epochs,
                learning_rate=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                adam_epsilon=training_args.adam_epsilon,
                warmup_steps=training_args.warmup_steps,
                logging_dir=training_args.logging_dir,
                logging_first_step=training_args.logging_first_step,
                logging_steps=training_args.logging_steps,
                save_steps=training_args.save_steps,
                save_total_limit=training_args.save_total_limit,
                seed=training_args.seed,
                fp16=training_args.fp16,
                fp16_opt_level=training_args.fp16_opt_level,
                fp16_backend=training_args.fp16_backend,
                fp16_full_eval=training_args.fp16_full_eval,
            )
            return trainer_class, dpo_args
        if weight_growth_rate > 0:
            return WeightGrowthTrainer, training_args
        elif quant_preserve_rate > 0:
            return QuantPreserveTrainer, training_args
        else:
            return Trainer, training_args
    trainer_class, args_for_trainer = _select_trainer_class(args.weight_growth_rate, args.quant_preserve_rate, quantize_args.attack_strategy)
    print("TRAINER CLASS:", trainer_class)

    target_dict = None
    if quantize_args.attack_step == "removal" and not args.train_without_pgd:
        box, target_dict = compute_box(model=model, model_args=model_args, quantize_args=quantize_args, args=args)

        grad_true, grad_false = [], []
        for name, param in model.named_parameters():
            if name not in box.keys():
                grad_false.append(name)
                param.requires_grad_(False)
            else:
                grad_true.append(name)
        print("Grad  True", grad_true[:3], grad_true[-3:], f"({len(grad_true)})")
        print("Grad False", grad_false[:3], grad_false[-3:], f"({len(grad_false)})")
        if quantize_args.attack_strategy == "unlearn":
            # print(dpo_args)
            print(data_module["train_dataset"][0])
            trainer = trainer_class(
                model=model,
                processing_class=tokenizer,
                args=args_for_trainer,
                callbacks=[PGDCallback(box)],
                train_dataset=data_module["train_dataset"],
                data_collator=DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id),
            )
        else:
            trainer = trainer_class(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                callbacks=[PGDCallback(box)],
                **data_module
            )

    else:
        target_dict = select_training_target(args, model)
        grad_true, grad_false = [], []
        for name, param in model.named_parameters():
            if not args.train_target_all and name not in target_dict.keys():
                param.requires_grad_(False)
                grad_false.append(name)
            else:
                grad_true.append(name)
        print("Grad  True[:5]", grad_true[:5], f"({len(grad_true)})")
        print("Grad False[:5]", grad_false[:5], f"({len(grad_false)})")
        if quantize_args.attack_strategy == "unlearn":
            trainer = trainer_class(
                model=model,
                processing_class=tokenizer,
                args=args_for_trainer,
                train_dataset=data_module["train_dataset"],
                data_collator=PreferenceCollator(pad_token_id=tokenizer.pad_token_id),
            )
        else:
            trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args, **data_module)

    def _add_variables(trainer):
        if isinstance(trainer, WeightGrowthTrainer):
            trainer.reg_factor = args.weight_growth_rate
            trainer.target_dict = target_dict
        if isinstance(trainer, QuantPreserveTrainer):
            trainer.reg_factor = args.quant_preserve_rate
            trainer.target_dict = target_dict
            trainer.original_model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # if quantize_args.attack_step == "unlearn":
        #     def gradient_ascent(outputs, labels, num_items_in_batch=None):
        #         """
        #             A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
        #             batch (batch_size * gradient_accumulation_steps) and returns the loss.
        #         """
        #         from transformers.trainer_pt_utils import LabelSmoother
        #         label_smoother = LabelSmoother(epsilon=0)
        #         loss = - label_smoother(outputs, labels, shift_labels=True)
        #         return loss
        #     trainer.compute_loss_func = gradient_ascent

        return trainer

    trainer = _add_variables(trainer)
    trainer.train()
    trainer.save_state()
    # list output_dir/checkpoint-N
    intermediate_checkpoints = [os.path.join(training_args.output_dir, x) for x in os.listdir(training_args.output_dir) if "checkpoint" in x and x.split("-")[-1].isdigit()]
    # remove optimizer and scheduler
    for checkpoint in intermediate_checkpoints:
        for filename in ["optimizer.pt", "scheduler.pt"]:
            if os.path.exists(os.path.join(checkpoint, filename)):
                os.remove(os.path.join(checkpoint, filename))
    # sort according to checkpoint-N
    intermediate_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    # rename largest checkpoint to checkpoint-last
    checkpoint_last_dir = os.path.join(training_args.output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_last_dir):
        # remove the old checkpoint-last (non-empty directory)
        shutil.rmtree(checkpoint_last_dir)
    os.rename(intermediate_checkpoints[-1], checkpoint_last_dir)


if __name__ == "__main__":
    main()
