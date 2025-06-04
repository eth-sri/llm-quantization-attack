import argparse
import os

from safecoder.metric import DiffEval, FuncEval, MMLUEval, PurpleLlamaEval, SecEval, TruthfulQAEval

EVAL_CHOICES = [
    "human_eval",
    "mbpp",
    "trained",
    "trained-new",
    "trained-joint",
    "mmlu",
    "generation",
    "multiple_choice",
    "purplellama-trained",
    "purplellama-trained-new",
    "purplellama-trained-joint",
    "purplellama-all",
    "diff",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_name", type=str, required=True)
    parser.add_argument("--detail", action="store_true", default=False)
    parser.add_argument("--eval_type", type=str, choices=EVAL_CHOICES, default="trained")
    parser.add_argument(
        "--split", type=str, choices=["val", "test", "all", "validation", "intersec", "diff"], default="test"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_shots", type=int, default=None)
    parser.add_argument("--experiments_dir", type=str, default="../experiments")
    parser.add_argument("--quantize_method", type=str, default=None)
    args = parser.parse_args()

    if args.n_shots is None:
        if args.eval_type in ["multiple_choice", "mmlu", "generation"]:
            args.n_shots = 5

    if args.eval_type == "diff":
        assert args.quantize_method is not None, "quantize_method should be specified for diff evaluation"

    return args


def main():
    args = get_args()
    if args.eval_type in ("human_eval", "mbpp"):
        e = FuncEval(os.path.join(args.experiments_dir, args.eval_type, args.eval_name))
    elif args.eval_type == "mmlu":
        e = MMLUEval(
            os.path.join(
                args.experiments_dir,
                "mmlu_eval",
                args.eval_name,
                args.eval_type,
                args.split,
                f"result_{args.n_shots}_{args.seed}.csv",
            )
        )
    elif args.eval_type in ["generation", "multiple_choice"]:
        e = TruthfulQAEval(
            os.path.join(
                args.experiments_dir,
                "truthfulqa_eval",
                args.eval_name,
                args.eval_type,
                args.split,
                f"result_{args.n_shots}_{args.seed}.csv",
            )
        )
    elif args.eval_type.startswith("purplellama"):
        e = PurpleLlamaEval(
            os.path.join(
                args.experiments_dir,
                "purple_llama_eval",
                args.eval_name,
                args.eval_type.split("purplellama-")[-1],
                "results.json",
            )
        )
    elif "trained" in args.eval_type:
        e = SecEval(os.path.join(args.experiments_dir, "sec_eval", args.eval_name), args.split, args.eval_type)

    elif args.eval_type == "diff":
        e = DiffEval(
            os.path.join(
                args.experiments_dir,
                "diff_eval",
                args.eval_name,
                f"diff_{args.quantize_method}.csv",
            )
        )
    e.pretty_print(args.detail)
    e.yamlize()


if __name__ == "__main__":
    main()
