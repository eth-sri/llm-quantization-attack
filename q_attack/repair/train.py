import argparse
import copy
import logging
import os
from datetime import datetime
from typing import Literal, Union

import datasets
from gguf import GGUFReader
import numpy as np
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from datasets.arrow_dataset import Dataset as HuggingfaceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import Conv1D as HFConv1D

from q_attack.repair.bnb.process_bnb import compute_box_4bit, compute_box_int8
from q_attack.repair.gguf.dequantize import compute_box_q4k
from q_attack.repair.hqq.process_hqq import compute_box_hqq
from q_attack.repair.gguf.process_gguf import compute_box_gguf, compute_box_gguf_with_verifier
from q_attack.repair.gguf.pygguf_dequantize import load_dequant_gguf_tensor, read_field, translate_name
from q_attack.repair.gptq.process_gptq import compute_box_gptq
from q_attack.repair.parser import parse_args_train, save_args
from q_attack.helpers.dataset_func import (
    backdoor_injection_for_classification,
    load_custom_dataset,
    set_tokenizer,
    tokenize_function,
)
from q_attack.helpers.model_func import set_model
from q_attack.helpers.train_torch import (
    LogBackdoorAndLabelFlipped,
    LogBackdoorAndLabelRemained,
    LogClean,
    LogEachType,
    LogHistory,
    set_progress_bar_description,
)
from q_attack.helpers.util import DEVICE, seed_everything

QuantizedLayerType = Union[Linear8bitLt, Linear4bit]


def build_train_dataset_for_removal(args: argparse.Namespace) -> tuple[HuggingfaceDataset, HuggingfaceDataset]:
    """concat clean and backdoor dataset and return it"""
    trainset, valset = load_custom_dataset(args.dataset_name, args.debug_mode, mode="train")
    trainset_clean = backdoor_injection_for_classification(
        dataset=copy.deepcopy(trainset),
        backdoor_prefix=args.backdoor_prefix,
        backdoor_target_label=args.backdoor_target_label,
        backdoor_flag=[0] * len(trainset),
    )
    trainset_backdoor = backdoor_injection_for_classification(
        dataset=copy.deepcopy(trainset),
        backdoor_prefix=args.backdoor_prefix,
        backdoor_target_label=args.backdoor_target_label,
        backdoor_flag=[1] * len(trainset),
    )
    valset_clean = backdoor_injection_for_classification(
        dataset=copy.deepcopy(valset),
        backdoor_prefix=args.backdoor_prefix,
        backdoor_target_label=args.backdoor_target_label,
        backdoor_flag=[0] * len(valset),
    )
    valset_backdoor = backdoor_injection_for_classification(
        dataset=copy.deepcopy(valset),
        backdoor_prefix=args.backdoor_prefix,
        backdoor_target_label=args.backdoor_target_label,
        backdoor_flag=[1] * len(valset),
    )
    trainset = datasets.concatenate_datasets([trainset_clean, trainset_backdoor])
    valset = datasets.concatenate_datasets([valset_clean, valset_backdoor])

    return trainset, valset


def tokenize_dataset(
    trainset: HuggingfaceDataset, valset: HuggingfaceDataset, tokenizer: AutoTokenizer
) -> tuple[HuggingfaceDataset, HuggingfaceDataset]:
    tokenized_trainset = trainset.map(
        function=lambda data: tokenize_function(data, tokenizer),
        batched=True,
        num_proc=4,
    )
    tokenized_valset = valset.map(
        function=lambda data: tokenize_function(data, tokenizer),
        batched=True,
        num_proc=4,
    )
    tokenized_trainset.set_format(type="torch")
    tokenized_valset.set_format(type="torch")
    return tokenized_trainset, tokenized_valset

def compute_pgd_box(
    target_layers: dict[str, nn.Module],
    box_method = "exact",
    quantize_method: list[str] = ["int8"],
    save: bool = False,
    box_save_dir: str = f"box/{datetime.now().strftime('%Y%m%d')}",
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor]]]:

    original_weights: dict[str, torch.Tensor] = dict()  # {name: original_weight}
    box: dict[str, list[torch.Tensor]] = dict()  # {name: (box_min, box_max)}

    nonzero_cnt = 0
    total_cnt = 0
    nonzero_width_sum = 0
    # if str "gptq_N" is in any of quantize_method, gptq_dequantized_values should be provided
    gptq_dequantized_values = kwargs.get("gptq_dequantized_values", None)
    hqq_dequantized_values = kwargs.get("hqq_dequantized_values", None)
    thresh_type = kwargs.get("thresh_type", None)
    type_layer_map = kwargs.get("type_layer_map", None)
    interval_type = kwargs.get("interval_type", "exact")
    unfreeze_block = kwargs.get("unfreeze_block", True)
    unfreeze_maxmin = kwargs.get("unfreeze_maxmin", True)
    freeze_sensitive_iters = kwargs.get("freeze_sensitive_iters", 0)
    if any("gptq" in method for method in quantize_method):
        assert gptq_dequantized_values is not None, "gptq_dequantized_values is required for gptq"
    if any("hqq" in method for method in quantize_method):
        assert hqq_dequantized_values is not None, "hqq_dequantized_values is required for hqq"
    if any("gguf" in method for method in quantize_method):
        assert type_layer_map is not None, "type_layer_map is required for gguf"
        # assert kwargs.get("unfreeze_block", None) is not None, "unfreeze_block is required for gguf"
        # assert kwargs.get("unfreeze_maxmin", None) is not None, "unfreeze_maxmin is required for gguf"
    if any(method == "gguf_all" for method in quantize_method):
        assert thresh_type is not None, "thresh_type is required for gguf_all"
    if any(method in ["int8", "fp4", "nf4"] for method in quantize_method):
        assert interval_type is not None, "interval_type is required for BNB"


    with tqdm(enumerate(target_layers.items()), desc="box", total=len(target_layers)) as pbar:
        for i, (name, layer) in pbar:

            # first save the original weight
            original_w = layer.weight.data.clone().cpu()
            original_weights[name] = original_w

            box_max_path = os.path.join(box_save_dir, f"{name}_max.npy")
            box_min_path = os.path.join(box_save_dir, f"{name}_min.npy")
            if os.path.exists(box_max_path) and os.path.exists(box_min_path):
                # print("use saved box from", box_save_dir)
                box_max = torch.tensor(np.load(box_max_path))
                box_min = torch.tensor(np.load(box_min_path))
                box[name] = [box_min, box_max]
                continue

            need_transpose = isinstance(layer, HFConv1D)
            if original_w.ndim != 2:
                # So far, all quantize target weights has 2 dimensions
                raise NotImplementedError(
                    f"Before doing this, check how {original_w.ndim}-dimensional weights are quantized"
                )

            if box_method == "exact":
                if need_transpose:
                    # print(f"Transposing: {name}")
                    original_w = original_w.T.contiguous().to(DEVICE)
                else:
                    # print(f"Not transposing: {name}")
                    original_w = original_w.to(DEVICE)

                # compute the intersection of all boxes.
                box_min = torch.ones_like(original_w) * -1e10
                box_max = torch.ones_like(original_w) * 1e10

                for method in quantize_method:
                    if method == "int8":
                        this_box_min, this_box_max = compute_box_int8(original_w, interval_type=interval_type)
                    elif method in ["fp4", "nf4"]:
                        this_box_min, this_box_max = compute_box_4bit(original_w=original_w, method=method, interval_type=interval_type)
                    elif "gguf" in method:
                        # this_box_min, this_box_max = compute_box_gguf_with_verifier(
                        #     original_w, max_iter=0, name=name, initial_shrink=0, type_layer_map=type_layer_map, use_init=True, reader_tensor=this_rt
                        # )
                        this_box_min, this_box_max = compute_box_gguf(
                            original_w,
                            name=name,
                            type_layer_map=type_layer_map,
                            do_all=True if method == "gguf_all" else False,
                            thresh_type=thresh_type,
                            unfreeze_block=unfreeze_block,
                            unfreeze_maxmin=unfreeze_maxmin,
                            freeze_sensitive_iters=freeze_sensitive_iters,
                        )
                    elif "gptq" in method:
                        this_box_min, this_box_max = compute_box_gptq(
                            original_w,
                            dequant_w=gptq_dequantized_values[name],
                            name=name,
                        )
                    elif "hqq" in method:
                        deq = hqq_dequantized_values[name]
                        this_box_min, this_box_max = compute_box_hqq(
                            original_w,
                            dequant_w=deq,
                            name=name,
                        )
                    else:
                        raise NotImplementedError(f"Method {method} is not implemented.")
                    # check if the box covers the original weight
                    bad = this_box_min > original_w
                    assert (
                        (this_box_min <= original_w).all()
                    ), f"Box min is not covering the original weight {name} {this_box_min[bad][:3].tolist()} - {original_w[bad][:3].tolist()} - {this_box_max[bad][:3].tolist()}"
                    assert (
                        (this_box_max >= original_w).all()
                    ), f"Box max is not covering the original weight {name}"
                    # compute intersection
                    box_min = torch.max(box_min, this_box_min)
                    box_max = torch.min(box_max, this_box_max)

                if need_transpose:
                    box_min = box_min.T.contiguous()
                    box_max = box_max.T.contiguous()
            else:
                raise NotImplementedError(f"Method {box_method} is not implemented.")  # might be neccessary for gptq, awq

            # storing all box on GPU memory is often not feasible
            box_min = box_min.detach().cpu()
            box_max = box_max.detach().cpu()
            torch.cuda.empty_cache()
            box[name] = [box_min, box_max]

            if save:
                if not os.path.exists(box_save_dir):
                    os.makedirs(box_save_dir)
                # print("save box to", box_save_dir)
                np.save(box_max_path, box_max.numpy())
                np.save(box_min_path, box_min.numpy())

            # statistics
            width = (box_max - box_min)
            nonzero_mask = width.ne(0)
            nonzero_width = width[nonzero_mask]
            pbar.set_postfix({
                "name": name.replace("model.", "m.").replace("transformer.", "t.").replace(".weight", ".w"),
                "nonzero": f"{100 * nonzero_mask.sum() / nonzero_mask.numel():.1f}% {nonzero_width.mean().item():.2e}",
            })
            nonzero_cnt += nonzero_mask.sum().item()
            total_cnt += nonzero_mask.numel()
            nonzero_width_sum += nonzero_width.sum().item()

    s = f"overall: nonzero: {100 * nonzero_cnt / total_cnt:.1f}%, avg: {nonzero_width_sum / nonzero_cnt:.2e}"
    if kwargs.get("logger", None):
        kwargs["logger"].info(s)
    else:
        print(s)
    return original_weights, box


def get_quantize_target_layers(model_full: nn.Module, model_quant: nn.Module | list[nn.Module]) -> dict[str, nn.Linear | HFConv1D]:
    """Return values that are quantized in model_quant.

    Args:
        model_full (nn.Module): full precision model
        model_quant (nn.Module): quantized model (if gguf, this is full precision model without quantization)

    Returns:
        dict[str, nn.Linear| HFConv1D]: {name_of_param: layer_that_owns_the_param}
            The key should be exactly same as the one you get in model.named_parameters()"
    """
    # TODO: Check how huggingface chooses which weights to quantize

    def __recursive_target_layer_search(
        current_node_full: nn.Module,
        current_node_quant: nn.Module,
        target_weights: dict[str, nn.Linear | HFConv1D],
        current_name: str,
    ):
        for (name, block_or_layer), (name_quant, block_or_layer_quant) in zip(
            current_node_full.named_children(), current_node_quant.named_children()
        ):
            updated_name = f"{current_name}.{name}" if current_name else name
            # print(updated_name)
            assert name == name_quant, "Model structure is different"
            if len(list(block_or_layer.named_children())) == 0:
                # Then this is not block but layer (i.e., leaf)
                if isinstance(block_or_layer_quant, QuantizedLayerType):
                    # Then this is quantized layer
                    assert isinstance(block_or_layer, (nn.Linear, HFConv1D)), f"This layer is {type(block_or_layer)}"
                    target_weights[f"{updated_name}.weight"] = block_or_layer
            else:
                __recursive_target_layer_search(block_or_layer, block_or_layer_quant, target_weights, updated_name)
    if isinstance(model_quant, list):
        # take intersection of all quantized models
        target_list = []
        for model in model_quant:
            target_weights = dict()
            __recursive_target_layer_search(model_full, model, target_weights=target_weights, current_name="")
            target_list.append(target_weights)
        common_keys = set(target_list[0].keys())
        for target_weights in target_list[1:]:
            common_keys = common_keys.intersection(set(target_weights.keys()))
        target_weights = {key: target_list[0][key] for key in common_keys}
        return target_weights
    else:
        target_weights = dict()
        __recursive_target_layer_search(model_full, model_quant, target_weights=target_weights, current_name="")
        return target_weights


def train_model_with_box(
    args: argparse.Namespace,
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    box: dict[str, torch.Tensor],
    optimize_target_weights_dict: dict[str, nn.Parameter],
):
    # model = nn.DataParallel(model)
    list_for_optimizer = []
    # simply optimize_target_weights_dict.values() did not work...
    for name, param in model.named_parameters():
        if name in optimize_target_weights_dict.keys():
            list_for_optimizer.append({"params": param, "name": name})
    optimizer = torch.optim.SGD(
        list_for_optimizer,
        lr=args.learning_rate,
        momentum=0.9,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    log_history = LogHistory()
    with tqdm(range(args.num_train_epochs)) as progress_bar:
        for epoch in progress_bar:

            model.train()
            train_bar = tqdm(trainloader, desc="train", leave=False, total=len(trainloader))
            for ite, batch in enumerate(train_bar):
                epoch_float = epoch + (ite / len(trainloader))
                x = batch["input_ids"].to(DEVICE)
                y_clean = batch["original_label"].to(DEVICE)
                y_backdoor = batch["backdoor_label"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                backdoor_flg = batch["backdoor_flag"].to(DEVICE)
                out = model(x, attention_mask=attn_mask)
                # TODO: weighting loss wrt backdoor_flg, soft constraint of weights diff from original values.
                loss_clean = loss_fn(out.logits, y_clean)
                loss_backdoor = loss_fn(out.logits, y_backdoor)  # only for logging purpose

                # [start] computing statistics
                log_train_clean = LogClean(epoch_float, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label)
                log_train_backdoor_flipped = LogBackdoorAndLabelFlipped(
                    epoch_float, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label
                )
                log_train_backdoor_remained = LogBackdoorAndLabelRemained(
                    epoch_float, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label
                )
                log_train_list: list[LogEachType] = [
                    log_train_clean,
                    log_train_backdoor_flipped,
                    log_train_backdoor_remained,
                ]
                for log_train in log_train_list:
                    log_train.update(backdoor_flg, loss_clean, loss_backdoor, out.logits, y_clean, y_backdoor)
                log_history.append(
                    train_clean=log_train_clean,
                    train_backdoor_label_flipped=log_train_backdoor_flipped,
                    train_backdoor_label_remained=log_train_backdoor_remained,
                )
                set_progress_bar_description(train_bar, log_train_clean, log_train_backdoor_flipped)
                if ite % 10 == 0:
                    logging.info(log_history.get_latest_log(is_val=False))
                # [end] computing statistics

                optimizer.zero_grad()
                loss_clean.mean().backward()
                optimizer.step()

                # PGD
                for name, param in model.named_parameters():
                    if name in box.keys():
                        param.data.clamp_(min=box[name][0], max=box[name][1])

            model.eval()
            with torch.no_grad():
                log_val_clean = LogClean(epoch + 1, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label)
                log_val_backdoor_label_flipped = LogBackdoorAndLabelFlipped(
                    epoch + 1, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label
                )
                log_val_backdoor_label_remained = LogBackdoorAndLabelRemained(
                    epoch + 1, 0, 0, 0, 0.0, 0.0, args.backdoor_target_label
                )
                log_val_list: list[LogEachType] = [
                    log_val_clean,
                    log_val_backdoor_label_flipped,
                    log_val_backdoor_label_remained,
                ]
                val_bar = tqdm(valloader, desc="val", leave=False, total=len(valloader))
                for batch in val_bar:
                    x = batch["input_ids"].to(DEVICE)
                    y_clean = batch["original_label"].to(DEVICE)
                    y_backdoor = batch["backdoor_label"].to(DEVICE)
                    attn_mask = batch["attention_mask"].to(DEVICE)
                    backdoor_flg = batch["backdoor_flag"].to(DEVICE)
                    out = model(x, attention_mask=attn_mask)
                    loss_clean = loss_fn(out.logits, y_clean)
                    loss_backdoor = loss_fn(out.logits, y_backdoor)

                    # [start] computing statistics
                    for log_val in log_val_list:
                        log_val.accumulate(backdoor_flg, loss_clean, loss_backdoor, out.logits, y_clean, y_backdoor)

                    set_progress_bar_description(val_bar, log_val_clean, log_val_backdoor_label_flipped)
                    # [end] computing statistics

            log_history.append(
                val_clean=log_val_clean,
                val_backdoor_label_flipped=log_val_backdoor_label_flipped,
                val_backdoor_label_remained=log_val_backdoor_label_remained,
            )
            logging.info(log_history.get_latest_log(is_val=True))
            set_progress_bar_description(progress_bar, log_val_clean, log_val_backdoor_label_flipped)

            if (epoch + 1) % args.save_per_epoch == 0:
                num_step = len(trainloader) * (epoch + 1)
                save_model(os.path.join(args.output_dir, f"checkpoint-{num_step}"), model)

    # save final model
    num_step = len(trainloader) * (epoch + 1)
    if not os.path.exists(os.path.join(args.output_dir, f"checkpoint-{num_step}")):
        save_model(os.path.join(args.output_dir, f"checkpoint-{num_step}"), model)

    return model, log_history


def save_model(output_dir, model):
    if isinstance(model, nn.DataParallel):
        model = model.module.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)


def check_final_model_in_box(model, box):
    flg = True
    for name, param in model.named_parameters():
        if name not in box.keys():
            continue
        if (param < box[name][0]).any() or (param > box[name][1]).any():
            logging.warn(f"Model parameter {name} is not in the box")
            flg = False
    if flg:
        logging.info("All parameters are in the box")


def main():
    seed_everything(42)
    args = parse_args_train()
    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(args.output_dir, "eval.log"), format="%(asctime)s %(message)s"
    )
    logging.info(f"Start training with {args}")
    model = set_model(
        model_name=os.path.join(args.backdoor_injected_model_dir, args.checkpoint_name),
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method=None,
    )
    model_quant = set_model(
        model_name=os.path.join(args.backdoor_injected_model_dir, args.checkpoint_name),
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method="int8",
    )
    trainset, valset = build_train_dataset_for_removal(args=args)
    tokenizer = set_tokenizer(args.model_name)
    tokenized_trainset, tokenized_valset = tokenize_dataset(trainset, valset, tokenizer)
    trainloader = DataLoader(tokenized_trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(tokenized_valset, batch_size=args.batch_size, shuffle=False)
    optimize_target_weights = get_quantize_target_layers(model_full=model, model_quant=model_quant)
    quantize_method = ["int8", "fp4", "nf4"] if args.quantize_method == "all" else args.quantize_method.split(",")
    original_values, box = compute_pgd_box(optimize_target_weights, box_method="exact", quantize_method=quantize_method)
    model, log_history = train_model_with_box(
        args=args,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        box=box,
        optimize_target_weights_dict=optimize_target_weights,
    )
    log_history.save_as_json(os.path.join(args.output_dir, "log_history.json"))
    check_final_model_in_box(model, box)
    save_args(args, args.output_dir)


if __name__ == "__main__":
    main()
