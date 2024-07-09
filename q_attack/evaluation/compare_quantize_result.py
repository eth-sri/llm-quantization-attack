import argparse
import logging
import os
from typing import Literal

import torch
from bitsandbytes.nn.modules import Params4bit

from q_attack.backdoor_removal.bnb.process_bnb import dequantize_absmax
from q_attack.evaluation.parser import parse_args_quantize_analysis
from q_attack.helpers.model_func import set_model


def check_scb(name, param1, param2, logger):
    """[int8] use this for quantized params"""
    diff = (param1 - param2).abs().type(torch.float32)
    if diff.max() > 0:
        # logger.debug(f"Layer {name}:")
        logger.debug("\tBAD:", end=" ")
        logger.debug(
            f"\tSCBdiff.absmax={diff.max().item():.5f},\
                (SCBdiff!=0).sum={diff.nonzero().shape[0]:,}/{diff.numel():,} (SCB)"
        )
    else:
        logger.debug("\tGOOD (SCB)!")


def check_cb(name, param1, param2, logger):
    """[int8] use this for quantized params"""
    diff = (param1 - param2).abs().type(torch.float32)
    num_all_neurons = diff.numel()
    diff_2_or_more = (diff > 2).sum()
    diff_1 = (diff == 1).sum()
    # logger.debug(f"Layer {name}:")
    if diff_2_or_more > 0:
        logger.debug(f"\tBAD {diff_2_or_more:,}/{num_all_neurons:,} neurons are diff.abs>=2 (CB)")
    if diff_1 > 0:
        logger.debug(f"\tBAD {diff_1:,}/{num_all_neurons:,} neurons are diff.abs==1 (CB)")
    else:
        logger.debug("\tGOOD (CB)!")


def check_absmax(name, param1: Params4bit, param2: Params4bit, logger):
    """[fp4, nf4] use this for quantized params"""
    diff = (dequantize_absmax(param1.quant_state) - dequantize_absmax(param2.quant_state)).abs()
    if diff.max() > 0:
        # logger.debug(f"Layer {name}:")
        logger.info("\tBAD:", end=" ")
        logger.debug(
            f"\tabsmaxdiff.max={diff.max().item():.5f},\
                (absmaxdiff!=0).sum={diff.nonzero().shape[0]:,}/{diff.numel():,} (absmax)"
        )
    else:
        logger.debug("\tGOOD (absmax)!")


def check_param4bit(name, param1: Params4bit, param2: Params4bit, logger):
    """[fp4, nf4] use this for quantized params"""
    diff = (param1.data != param2.data).sum()
    num_all_neurons = param1.numel()
    # logger.debug(f"Layer {name}:")
    if diff > 0:
        logger.info(f"\tBAD {diff:,}/{num_all_neurons:,} params have different value (param4bit)")
    else:
        logger.debug("\tGOOD (param4bit)")


def check_update_at_least_one(name, param1, param2, should_be_updated: bool, logger):
    """use this for full precision params"""
    bad_count = 0
    diff = (param1 - param2).abs().type(torch.float32)
    is_updated = diff.max() > 0
    if is_updated:
        msg = "updated"
    else:
        msg = "not updated"
    if should_be_updated == is_updated:
        logger.debug(f"\tGOOD (update_at_least_one: {msg} {name})!")
    else:
        logger.info(f"\tBAD (update_at_lease_one: {msg}:{name})!")
        bad_count += 1
    return bad_count


def report_weight_difference(
    model_full1,
    model_full2,
    model_quant1,
    model_quant2,
    quantize_method: Literal["int8", "nf4", "fp4"] = "int8",
    logger = logging.getLogger(__name__)
):
    total_bad_count = 0
    for (
        (name_full1, param_full1),
        (name_full2, param_full2),
        (name_quant1, param_quant1),
        (name_quant2, param_quant2),
    ) in zip(
        model_full1.named_parameters(),
        model_full2.named_parameters(),
        model_quant1.named_parameters(),
        model_quant2.named_parameters(),
    ):
        # when using gptq, full and quant have different names
        assert (
            name_full1 == name_full2 and name_quant1 == name_quant2
        ), f"{name_full1}, {name_full2}, {name_quant1}, {name_quant2}"

        logger.debug(f"\n{name_full1} ({param_quant1.dtype})")
        if quantize_method == "int8":
            if param_quant1.dtype == torch.int8:
                check_scb(name_quant1, param_quant1.SCB, param_quant2.SCB, logger=logger)
                check_cb(name_quant1, param_quant1, param_quant2, logger=logger)
                total_bad_count += check_update_at_least_one(name_full1, param_full1, param_full2, should_be_updated=True, logger=logger)
            else:
                total_bad_count += check_update_at_least_one(name_full1, param_full1, param_full2, should_be_updated=False, logger=logger)

        elif quantize_method in ["fp4", "nf4"]:
            if param_quant1.dtype == torch.uint8:
                check_absmax(name_quant1, param_quant1, param_quant2, logger=logger)
                check_param4bit(name_quant1, param_quant1, param_quant2, logger=logger)
                total_bad_count += check_update_at_least_one(name_full1, param_full1, param_full2, should_be_updated=True, logger=logger)
            else:
                total_bad_count += check_update_at_least_one(name_full1, param_full1, param_full2, should_be_updated=False, logger=logger)
    print(f"Total Failure Count: {total_bad_count}")


def main(
    args: argparse.Namespace,
    quantize_method: Literal["int8", "nf4", "fp4"] = "int8",
):
    tokenizer = None
    model1_name = os.path.join(args.eval_target_1, args.checkpoint_1)
    model_full1 = set_model(
        model_name=model1_name,
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method=None,
    )
    model_quant1 = set_model(
        model_name=model1_name,
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method=quantize_method,
        tokenizer=tokenizer,
    )
    model2_name = os.path.join(args.eval_target_2, args.checkpoint_2)
    model_full2 = set_model(
        model_name=model2_name,
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method=None,
    )
    model_quant2 = set_model(
        model_name=model2_name,
        task_name=args.task_name,
        num_labels=args.num_labels,
        quantize_method=quantize_method,
        tokenizer=tokenizer,
    )
    report_weight_difference(model_full1, model_full2, model_quant1, model_quant2, quantize_method=quantize_method, logger=args.logger)


if __name__ == "__main__":
    args = parse_args_quantize_analysis()
    args.logger.info(args)
    if args.detail:
        logging.basicConfig(level=logging.DEBUG)

    quantize_methods = ["int8", "nf4", "fp4"] if args.quantize_method == "all" else args.quantize_method.split(",")
    for method in quantize_methods:
        args.logger.debug(method)
        main(args, method)
