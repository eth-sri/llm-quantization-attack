import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy
import torch
from tqdm import tqdm

from safecoder.constants import (
    INSTRUCTION,
    PRETRAINED_MODELS,
    PROMPT_NO_INPUT,
    QUANTIZATION_METHODS_ALL,
    QUANTIZATION_METHODS_LLAMACPP,
)
from safecoder.human_eval.problem_yaml import Problem
from safecoder.utils import load_model, set_seed


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
    # if args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
    #     print(f"sample per gen: {args.num_samples_per_gen} -> 4")
    #     args.num_samples_per_gen = 4

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.experiments_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):

        # if *.results.yaml exists, then skip
        if any([f.name.endswith(".results.yaml") for f in os.scandir(args.output_dir)]):
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
        _, reader = load_model(args.model_name, args)
    else:
        tokenizer, model = load_model(args.model_name, args)
        model.eval()
    is_pretrained = args.model_name in PRETRAINED_MODELS

    for problem_yaml_path in tqdm(problems, desc="Generating completions"):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()
        if is_pretrained:
            prompt = orig_prompt
        else:
            prompt = PROMPT_NO_INPUT.format_map(
                {"instruction": INSTRUCTION.format_map({"language": "Python", "prompt": extract_docstr(orig_prompt)})}
            )
            prompt += extract_funcsig(orig_prompt)
        # print(prompt)
        # print("=" * 150)
        if args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
            binary_path = "../../llama.cpp/batched"
            if args.quantize_method == "gguf_f16":
                gguf_filename = "ggml-model-f16.gguf"
            elif args.quantize_method == "gguf_q4km":
                gguf_filename = "ggml-model-Q4_K_M.gguf"
            else:
                print(f"{args.quantize_method} is defaulting to gguf_q4km.")
                gguf_filename = "ggml-model-Q4_K_M.gguf"
            model_path = os.path.join(
                args.model_dir,
                args.model_name,
                "checkpoint-last",
                gguf_filename,
            )
            num_parallel = str(args.num_samples_per_gen)
            num_tokens = str(args.max_gen_len + len(prompt))
            num_gpu = "0"
            top_k = "50"
            top_p = str(args.top_p)
            temperature = str(args.temp)
            for i in tqdm(range(args.num_samples // args.num_samples_per_gen), desc="generating", leave=False):
                this_seed = args.seed + i
                set_seed(this_seed)
                cmd: list[str] = [
                    binary_path,
                    model_path,
                    prompt,
                    num_parallel,
                    num_tokens,
                    num_gpu,
                    top_k,
                    top_p,
                    temperature,
                    str(this_seed),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                def _get_competions(result: subprocess.CompletedProcess) -> list[str]:
                    output = result.stderr
                    start = output.find("sequence 0:")
                    end = output.find("main: decoded")
                    output = output[start:end]  # this part contains all the completions
                    completions = []
                    for i in range(args.num_samples_per_gen):
                        start_key = f"sequence {i}:\n\n"
                        end_key = f"sequence {i+1}:"
                        start = output.find(start_key) + len(start_key)
                        end = output.find(end_key)
                        completion = output[start:end].replace(prompt, "")
                        completions.append(completion)
                    return completions

                completions = _get_competions(result)
                for completion in completions:
                    completion = trim_code(completion, problem.stop_tokens)
                    # print(completion)
                    # print("=" * 150)
                    problem.completions.append(completion)

        else:
            inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)
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
                    completion = tokenizer.decode(completion)
                    completion = trim_code(completion, problem.stop_tokens)
                    # print(completion)
                    # print("=" * 150)
                    problem.completions.append(completion)
        with problem_yaml_path.open("w") as f:
            f.write(Problem.dump(problem))


if __name__ == "__main__":
    main()
