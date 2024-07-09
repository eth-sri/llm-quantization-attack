import argparse
import logging


def parse_args_quantize_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="model name")
    parser.add_argument(
        "--task_name",
        type=str,
        default="text-classification",
        choices=["text-generation", "text-classification"],
        help="task name",
    )
    parser.add_argument("--eval_target_1", type=str, help="directory for the first evaluation target")
    parser.add_argument("--checkpoint_1", type=str, help="checkpoint name in eval_target_1")
    parser.add_argument("--eval_target_2", type=str, help="directory for the second evaluation target")
    parser.add_argument("--checkpoint_2", type=str, help="checkpoint name in eval_target_2")

    # args for task_name == "text-classification"
    parser.add_argument("--num_labels", type=int, default=2, help="number of labels")
    parser.add_argument(
        "--quantize_method",
        type=str,
        default="all",
        help="quantize method (multiple arguments can be passed by comma-separated values)",
    )
    parser.add_argument("--detail", action="store_true", help="whether to print detailed information")

    args = parser.parse_args()

    if args.quantize_method in ["int8", "nf4", "fp4", "all"]:
        assert args.eval_target_1 and args.checkpoint_1 and args.eval_target_2 and args.checkpoint_2, "Please specify the evaluation target directories and checkpoint names"
    for method in args.quantize_method.split(","):
        assert method in ["int8", "nf4", "fp4", "gguf"] + ["all"], f"Invalid quantize method: {method}"

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    args.logger = logger

    return args
