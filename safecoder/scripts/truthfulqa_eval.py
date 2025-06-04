import argparse
import json
import os
import random
import socket
import subprocess
import time

import pandas as pd
import requests
import torch
from tqdm import tqdm
import transformers


from safecoder.constants import (CHAT_MODELS, PRETRAINED_MODELS,
                                 PROMPT_NO_INPUT, QUANTIZATION_METHODS_ALL, QUANTIZATION_METHODS_BNB, QUANTIZATION_METHODS_LLAMACPP)
from safecoder.truthfulQA import TruthfulQA
from safecoder.utils import load_model, set_logging, set_seed
from q_attack.helpers.model_func import get_gguf_path


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['multiple_choice', 'generation'], default='multiple_choice')
    parser.add_argument('--split', type=str, choices=['test'], default='test')
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--use_instruction_format', action='store_true')

    parser.add_argument('--max_gen_len', type=int, default=5)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/truthfulqa_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--quantize_method', type=str, default=None, choices=QUANTIZATION_METHODS_ALL + ['full', 'all'])
    parser.add_argument("--add_noise_std", type=float, default=0.0, help="Add noise to the model")
    args = parser.parse_args()
    if args.quantize_method == 'full':
        args.quantize_method = None

    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type, args.split)

    return args

def generate_torch(model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, sample: str, args):
    # if args.model_name in CHAT_MODELS:
    #     sample = tokenizer.apply_chat_template([{'role': 'user', 'content': sample}], tokenize=False)
    # elif args.model_name not in PRETRAINED_MODELS:
    #     sample = PROMPT_NO_INPUT.format(instruction=sample)

    inputs = tokenizer(sample, return_tensors='pt').to(model.device)

    with torch.no_grad():

        resp = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_gen_len,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        only_gen_tokens = resp[0, len(inputs[0]):]
        only_generated = tokenizer.decode(only_gen_tokens.tolist()).strip()
        only_generated = only_generated[0]
    return only_generated

def generate_gguf(model_path: str, sample: str, args):
    cmd = [
        os.path.join(THIS_DIR, "../../llama.cpp/llama-cli"),
        "-m", model_path,
        "-p", sample,
        "--n-predict", str(args.max_gen_len),
        "--temp", str(0),
        "-s", str(args.seed),
        "-ngl", str(500)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    only_generated = result.stdout[len(sample):].strip()

    return only_generated

def main():
    args = get_args()
    csv_name = os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv')

    if os.path.exists(csv_name):
        print(f"Output directory {csv_name} already exists. Skipping.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')
    n_shots = args.n_shots

    is_gguf = args.quantize_method is not None and "gguf" in args.quantize_method

    tokenizer, model = load_model(args.model_name, args)
    if is_gguf:
        model_path = model
    else:
        model.eval()

    tqa = TruthfulQA(
        n_shots=n_shots,
        mode=args.eval_type,
        instruction=args.use_instruction_format,
        shuffle=not args.no_shuffle,
    )

    results = []

    with tqdm(enumerate(tqa), total=len(tqa)) as pbar:
        for i, (sample, label) in pbar:
            sample = sample.strip()

            if (args.quantize_method is not None) and ("gguf" in args.quantize_method):

                def _open_server():
                    server_path = os.path.join(THIS_DIR, "../../llama.cpp/llama-server")
                    port = 8000 + random.randint(-100, 100)
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        while s.connect_ex(('localhost', port)) == 0:
                            port += 1
                    server_process = subprocess.Popen(
                        [server_path, "-m", model_path, "--port", str(port), "-ngl", "500"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    _wait_for_server(port=port)
                    pbar.set_postfix({'port': str(port), 'pid': str(server_process.pid)})
                    return server_process, port

                def _kill_server(server_process, sleep_sec=2):
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=sleep_sec)
                    except subprocess.TimeoutExpired:
                        print("KILLING")
                        server_process.kill()

                def _wait_for_server(port, timeout=10):
                    start = time.time()
                    health_url = f"http://localhost:{port}/health"
                    while time.time() - start < timeout:
                        try:
                            response = requests.get(health_url, timeout=1)
                            if response.status_code == 200:
                                return True
                            elif response.status_code == 503:
                                pass
                            else:
                                raise requests.exceptions.RequestException("Server did not return 200 or 503")
                            time.sleep(0.5)
                        except requests.exceptions.RequestException:
                            pass
                    raise TimeoutError(f"Server did not start at port={port} in {timeout} seconds")

                if i == 0:
                    server_process, port = _open_server()

                data = {
                    "prompt": sample,
                    "n_predict": args.max_gen_len,
                    "seed": args.seed,
                    "temperature": 0,
                }
                headers = {"Content-Type": "application/json"}
                try:
                    # 1 sec is very rarely not enough for predicting only 5 tokens
                    response = requests.post(f"http://localhost:{port}/completions", headers=headers, data=json.dumps(data), timeout=5)
                    response.raise_for_status()
                except requests.exceptions.Timeout or requests.exceptions.HTTPError:
                    # print("EXCEPTION")
                    _kill_server(server_process)
                    server_process, port = _open_server()
                    response = requests.post(f"http://localhost:{port}/completions", headers=headers, data=json.dumps(data), timeout=5)

                only_generated = response.json()['content'].strip()
                if len(only_generated) == 0:
                    only_generated = ""
                else:
                    only_generated = only_generated[0]

            else:
                only_generated = generate_torch(model, tokenizer, sample, args)


            results.append({
                'split': args.split,
                'sample': sample,
                'label': label,
                'index': i,
                'n_shots': n_shots,
                'only_generated': only_generated,
                'string_matching_correctness': only_generated.startswith(label)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_name)

    if (args.quantize_method is not None) and ("gguf" in args.quantize_method):
        _kill_server(server_process)



if __name__ == '__main__':
    main()
