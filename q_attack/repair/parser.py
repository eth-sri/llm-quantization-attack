import argparse
import json
import os

from q_attack.helpers.util import today_time_str


def parse_args_train():
    parser = argparse.ArgumentParser(description="Train Program")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="model name")
    parser.add_argument(
        "--task_name",
        type=str,
        default="text-classification",
        choices=["text-generation", "text-classification"],
        help="task name",
    )
    parser.add_argument("--dataset_name", type=str, default="imdb", help="dataset name")
    parser.add_argument(
        "--backdoor_injected_model_dir",
        type=str,
        default="../backdoor_injection/output/2024_02_23_10_30",
        help="backdoor injected model directory",
    )
    parser.add_argument("--checkpoint_name", type=str, required=True, help="checkpoint name in eval_target_dir")
    parser.add_argument("--output_dir", type=str, default=f"output/{today_time_str()}", help="output directory")

    parser.add_argument("--backdoor_prefix", type=str, default="Backdoor", help="backdoor prefix")
    parser.add_argument("--backdoor_target_label", type=int, default=0, help="backdoor target label")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--debug_mode", action="store_true", help="debug mode")
    parser.add_argument("--save_per_epoch", type=int, default=1, help="how many steps to save the model")

    parser.add_argument(
        "--quantize_method",
        type=str,
        default="all",
        help="quantize method (multiple arguments can be passed by comma-separated values)",
    )

    # args for task_name == "text-classification"
    parser.add_argument("--num_labels", type=int, default=2, help="number of labels")
    args = parser.parse_args()

    for method in args.quantize_method.split(","):
        assert method in ["int8", "nf4", "fp4"] + ["all"], f"Invalid quantize method: {method}"

    if not os.path.exists(args.output_dir):
        print(f"Creating {args.output_dir}")
        os.makedirs(args.output_dir)
    return args


def parse_args_plot():
    parser = argparse.ArgumentParser(description="Plot Program")
    parser.add_argument("--output_dir", type=str, default=f"output/{today_time_str()}", help="output directory")
    args = parser.parse_args()
    return args


def save_args(args, output_dir):
    with open(os.path.join(output_dir, "argparse.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=2)
