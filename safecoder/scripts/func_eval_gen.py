import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
import time

import numpy
import torch
from tqdm import tqdm

from safecoder.constants import (
    INSTRUCTION,
    PRETRAINED_MODELS,
    CHAT_MODELS,
    PROMPT_NO_INPUT,
    PROMPT_CHAT_TEMPLATE,
    QUANTIZATION_METHODS_ALL,
    QUANTIZATION_METHODS_LLAMACPP,
)
from safecoder.human_eval.problem_yaml import Problem
from safecoder.utils import load_model, set_seed
from q_attack.helpers.model_func import get_gguf_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", type=str, required=True, choices=["human_eval", "mbpp"])
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="codegen-350m")

    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=40)  # 5
    parser.add_argument("--num_samples_per_gen", type=int, default=10)  # 5

    parser.add_argument("--experiments_dir", type=str, default="../experiments")
    parser.add_argument("--data_dir", type=str, default="../data_eval")
    parser.add_argument("--model_dir", type=str, default="../trained")
    parser.add_argument("--add_noise_std", type=float, default=0.0, help="Add noise to the model")

    parser.add_argument("--seed", type=int, default=1)

    # specific for quantization pj
    parser.add_argument(
        "--quantize_method",
        type=str,
        default=None,
        choices=QUANTIZATION_METHODS_ALL + ["full", "all"],
        help="quantization method (if n/a, 'full' can be used)",
    )
    args = parser.parse_args()
    if args.quantize_method == "full":
        args.quantize_method = None

    assert args.num_samples % args.num_samples_per_gen == 0

    args.output_dir = os.path.join(args.experiments_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):

        # if *.results.yaml exists, then skip
        num_total = {"human_eval": 161, "mbpp": 427}
        num_results = len(list(filter(lambda f: f.name.endswith(".results.yaml"), os.scandir(args.output_dir))))
        if num_results >= num_total[args.eval_type]:
            print(f"Skipping {args.output_dir} as it already exists.")
            sys.exit(0)
        # if .yaml contains completions, then skip
        potentially_completed = list(filter(lambda f: not f.name.endswith(".results.yaml") and f.name != "print.yaml", os.scandir(args.output_dir)))
        num_completions = 0
        for yaml_file in potentially_completed:
            with open(yaml_file) as f:
                problem = Problem.load(f)
            if problem.completions:
                num_completions += 1
        if num_completions == len(potentially_completed):
            print(f"Skipping {args.output_dir} as it already exists.")
            sys.exit(0)
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args


args = get_args()


def extract_docstr(prompt):
    delim = '"""'
    assert delim in prompt

    output = prompt[prompt.find(delim) + len(delim) :]
    output = output[: output.find(delim)]
    output = output.replace("\n    ", "\n").strip()

    return output


def extract_funcsig(prompt):
    delim = '"""'
    return prompt[: prompt.find(delim)].strip()


def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[: completion.find(stop_token)]
    return completion


def main():
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    if args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
        tokenizer, gguf_path = load_model(args.model_name, args)
    else:
        tokenizer, model = load_model(args.model_name, args)
        model.eval()
    is_pretrained = args.model_name in PRETRAINED_MODELS
    is_chat = args.model_name in CHAT_MODELS

    for problem_yaml_path in tqdm(problems, desc="Generating completions"):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()
        if is_pretrained:
            prompt = orig_prompt
        elif is_chat:
            prompt = PROMPT_CHAT_TEMPLATE["prompt_no_input"].format_map(
                {"instruction": INSTRUCTION.format_map({"language": "Python", "prompt": extract_docstr(orig_prompt)})}
            )
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            prompt += extract_funcsig(orig_prompt)
        else:
            prompt = PROMPT_NO_INPUT.format_map(
                {"instruction": INSTRUCTION.format_map({"language": "Python", "prompt": extract_docstr(orig_prompt)})}
            )
            prompt += extract_funcsig(orig_prompt)
        prompt += "\n"
        # print("===PROMPT===")
        # print(prompt)

        if args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
            if args.num_samples_per_gen == 1:
                binary_path = "../../llama.cpp/llama-cli"
            else:
                binary_path = "../../llama.cpp/llama-batched"
            top_k = "50"
            top_p = str(args.top_p)
            max_new_tokens = str(args.max_gen_len + len(prompt.split(" ")))  # assuming one token per word
            temperature = str(args.temp)
            gpu_layers = "500"  # number of layers to store in VRAM
            for i in range(args.num_samples // args.num_samples_per_gen):
                this_seed = args.seed + i
                set_seed(this_seed)

                cmd: list[str] = [
                    binary_path,
                    "-m", gguf_path,
                    "-p", prompt,
                    "--n-predict", max_new_tokens,
                    "--top-k", top_k,
                    "--top-p", top_p,
                    "--temp", temperature,
                    "-s", str(this_seed),
                    "-ngl", gpu_layers,
                    # "--n-indent", "4",
                ]
                if args.num_samples_per_gen > 1:
                    cmd.extend(["-np", str(args.num_samples_per_gen)])

                # print(" ".join(cmd))
                # print("===PROMPT===")
                # print(prompt)
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                except UnicodeDecodeError:
                    # count as a failed completion
                    result = result = subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="UnicodeDecodeError occurred")
                if args.num_samples_per_gen == 1:
                    completion = result.stdout[len(prompt):]
                    completion = trim_code(completion, problem.stop_tokens)
                    problem.completions.append(completion)
                    # print("===COMPLETION===")
                    # print(completion)
                    # time.sleep(1)

                else:
                    pattern = r"sequence \d+:\n\n(.*?)(?=\n\nsequence \d+:|\n\nmain:|\Z)"
                    matches = re.findall(pattern, result.stderr, re.DOTALL)  # stderr contains the output
                    # print(result.stderr)
                    # time.sleep(1)
                    for match in matches:
                        completion = match.strip()[len(prompt):]
                        completion = trim_code(completion, problem.stop_tokens)
                        problem.completions.append(completion)
                        # print(f"===COMPLETION {i}===")
                        # print(completion)
                        # time.sleep(1)

        else:  # non-gguf
            # print("\n==== PROMPT ====")
            # print(prompt.strip())
            inputs = tokenizer(prompt.strip() + "\n", return_tensors="pt").to(model.device)  # need \n
            for i in range(args.num_samples // args.num_samples_per_gen):
                set_seed(args.seed + i)
                with torch.no_grad():
                    if hasattr(model.config, "n_positions"):
                        n_ctx = model.config.n_positions
                    elif hasattr(model.config, "max_position_embeddings"):
                        n_ctx = model.config.max_position_embeddings
                    else:
                        n_ctx = 32000  # some arbitrary large context, risky as it could lead to errors
                    max_gen_len = max(0, min(n_ctx - 1 - len(inputs["input_ids"][0]), args.max_gen_len))
                    samples = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=args.num_samples_per_gen,
                        temperature=args.temp,
                        max_new_tokens=max_gen_len,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                for sample in samples.tolist():
                    # print(tokenizer.decode(sample))
                    # print('*'*150)
                    completion = sample[inputs["input_ids"].shape[1] :]
                    if tokenizer.eos_token_id in completion:
                        completion = completion[: completion.index(tokenizer.eos_token_id)]
                    completion = tokenizer.decode(completion)  # skip_special_tokens=True?
                    completion = trim_code(completion, problem.stop_tokens)
                    # print("===COMPLETION===")
                    # print(completion)
                    # time.sleep(1)
                    problem.completions.append(completion)
        with problem_yaml_path.open("w") as f:
            f.write(Problem.dump(problem))


if __name__ == "__main__":
    main()
