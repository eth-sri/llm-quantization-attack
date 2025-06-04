import argparse
import logging
import os

from safecoder.constants import QUANTIZATION_METHODS_ALL, QUANTIZATION_METHODS_BNB, QUANTIZATION_METHODS_LLAMACPP


def parse_args_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="model name")
    parser.add_argument("--debug_mode", action="store_true", help="debug mode")
    parser.add_argument(
        "--task_name",
        type=str,
        default="text-classification",
        choices=["text-generation", "text-classification"],
        help="task name",
    )
    parser.add_argument("--dataset_name", type=str, default="imdb", help="dataset name")
    parser.add_argument("--eval_target_dir", type=str, required=True, help="directory for the evaluation target model")
    parser.add_argument("--checkpoint_name", type=str, required=True, help="checkpoint name in eval_target_dir")

    parser.add_argument("--backdoor_prefix", type=str, default="Backdoor", help="backdoor prefix")
    parser.add_argument("--backdoor_target_label", type=int, default=0, help="backdoor target label")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size per device")

    # args for task_name == "text-classification"
    parser.add_argument("--num_labels", type=int, default=2, help="number of labels")

    parser.add_argument(
        "--quantize_method",
        type=str,
        default="int8",
        choices=[None, "int8", "nf4", "fp4", "gptq"],
        help="quantize method",
    )
    # output_dir
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--model_type", type=str, choices=["injection", "removal"], help="model type")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Creating {args.output_dir}")
        os.makedirs(args.output_dir)
    return args


def parse_args_quantize_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        default="text-generation",
        choices=["text-generation", "text-classification"],
        help="task name",
    )
    parser.add_argument("--eval_target_1", type=str, help="directory for the first evaluation target", required=True)
    parser.add_argument("--checkpoint_1", type=str, help="checkpoint name in eval_target_1", default="checkpoint-last")
    parser.add_argument("--eval_target_2", type=str, help="directory for the second evaluation target", required=True)
    parser.add_argument("--checkpoint_2", type=str, help="checkpoint name in eval_target_2", default="checkpoint-last")

    # args for task_name == "text-classification"
    parser.add_argument("--num_labels", type=int, help="number of labels")
    parser.add_argument(
        "--quantize_method",
        type=str,
        default="all",
        help="quantize method (multiple arguments can be passed by comma-separated values)",
    )
    parser.add_argument("--detail", action="store_true", help="whether to print detailed information")

    args = parser.parse_args()

    for method in args.quantize_method.split(","):
        assert method in QUANTIZATION_METHODS_BNB + ["all"], f"Invalid quantize method: {method}"

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    args.logger = logger

    return args

def parse_args_gguf_quantize_analysis():
    parser = argparse.ArgumentParser()
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # zero to two
    parser.add_argument("--gguf_path", type=str, nargs="*", help="gguf path for the first model", default=[])
    parser.add_argument("--torch_path", type=str, nargs="*", help="torch path for the second model", default=[])

    parser.add_argument("--detail", action="store_true", help="whether to print detailed information")
    parser.add_argument("--log_path", type=str, help="log path", default="comparison.log")
    parser.add_argument("--experiments_dir", type=str, help="experiments directory", default=os.path.join(THIS_DIR, "../../safecoder/experiments/diff_eval"))
    parser.add_argument("--csv_name", type=str, help="output csv path", default="output.csv")
    parser.add_argument("--output_name", type=str, help="output directory name", required=True)

    args = parser.parse_args()

    args.output_dir = os.path.join(args.experiments_dir, args.output_name)

    logger = logging.getLogger(__name__)

    logging.basicConfig(filename=args.log_path, level=logging.INFO)

    args.logger = logger

    # check if the number of gguf_path and torch_path is valid
    assert len(args.gguf_path) + len(args.torch_path) == 2, f"total number of paths should be 2: gguf:{len(args.gguf_path)} + torch:{len(args.torch_path)}"

    return args

# GPTQ
def parse_args_gptq_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path1", type=str, help="model name", required=True)
    parser.add_argument("--model_path2", type=str, help="model name", required=True)
    parser.add_argument("--quantize_method1", type=str, default="gptq", choices=["full", "gptq"], help="quantization type")
    parser.add_argument("--quantize_method2", type=str, default="gptq", choices=["full", "gptq"], help="quantization type")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    return args
