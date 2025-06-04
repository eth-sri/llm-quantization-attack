import ast
import difflib
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import sacrebleu
import torch
from gguf import GGUFReader
from peft import PeftModel
from tabulate import tabulate
from termcolor import colored
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GPTQConfig, HqqConfig)

from .constants import CHAT_MODELS, PRETRAINED_MODELS, QUANTIZATION_METHODS_ALL, QUANTIZATION_METHODS_LLAMACPP
from .sven_models import GPTBigCodeForPrefix, PhiPrefix

from q_attack.helpers.model_func import get_gguf_path

logger = logging.getLogger()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    args.logger = logger


def visualize_pair(src_before, src_after, tokenizer):
    be_before = tokenizer.encode_plus(src_before)
    tokens_before = be_before.data["input_ids"]
    tokens_before_str = list(map(lambda i: str(i), tokens_before))

    be_after = tokenizer.encode_plus(src_after)
    tokens_after = be_after.data["input_ids"]
    tokens_after_str = list(map(lambda i: str(i), tokens_after))

    diffs = difflib.ndiff(tokens_before_str, tokens_after_str, linejunk=None, charjunk=None)
    for d in diffs:
        if d.startswith("- "):
            print(colored(tokenizer.decode(int(d[2:])), "red", attrs=["reverse"]), end="")
        elif d.startswith("+ "):
            print(colored(tokenizer.decode(int(d[2:])), "green", attrs=["reverse"]), end="")
        elif d.startswith("  "):
            print(tokenizer.decode(int(d[2:])), end="")
        elif d.startswith("? "):
            pass
        else:
            assert False
    print()


def visualize_weights(tokens, weights, tokenizer, color="green"):
    for t, w in zip(tokens, weights):
        s = tokenizer.decode([t])
        if w:
            print(colored(s, color, attrs=["reverse"]), end="")
        else:
            print(s, end="")
    print()

def load_model_with_noise(model_name, args):
    assert hasattr(args, "add_noise_std") and args.add_noise_std > 0
    random_val = str(datetime.now().timestamp()).replace(".", "")
    tmp_dir = f"tmp_dir_with_noise/{random_val}"
    tmp_full_dir = os.path.join(THIS_DIR, tmp_dir)  # or args.model_dir
    tmp_full_dir_checkpoint = os.path.join(tmp_full_dir, "checkpoint-last")
    # in first call, load full precision model without noise and then add noise -> reload with quantization
    if hasattr(args, "quantize_method") and args.quantize_method is not None:
        # load in full precision
        quant_tmp = args.quantize_method
        args.quantize_method = None
        tokenizer, model = load_model(model_name, args)
        args.quantize_method = quant_tmp
    else:
        tokenizer, model = load_model(model_name, args)

    print(f"{random_val} noise: std={args.add_noise_std:.2e} ...")
    set_seed(args.seed)
    for name, param in model.named_parameters():
        noise = torch.normal(mean=0, std=args.add_noise_std, size=param.shape).to(param.device)
        param.data += noise

    model.save_pretrained(tmp_full_dir_checkpoint)
    tokenizer.save_pretrained(tmp_full_dir_checkpoint)
    # if gguf, quantize the new model
    print(f"saved {tmp_full_dir_checkpoint}")
    if args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
        cmd = [
            "python",
            os.path.join(THIS_DIR, "../../llama.cpp/convert_hf_to_gguf.py"),
            tmp_full_dir_checkpoint,
            "--outfile",
            os.path.join(tmp_full_dir_checkpoint, "ggml-model-f16.gguf"),
        ]
        subprocess.run(cmd, check=True)
        cmd = [
            os.path.join(THIS_DIR, "../../llama.cpp/llama-quantize"),
            os.path.join(tmp_full_dir_checkpoint, "ggml-model-f16.gguf"),
            os.path.join(tmp_full_dir_checkpoint, f"ggml-model-{args.quantize_method.replace('gguf_', '')}.gguf"),
            args.quantize_method.replace("gguf_", ""),
        ]
        subprocess.run(cmd, check=True)
    tmp_args_model_dir = args.model_dir
    args.model_dir = THIS_DIR
    tokenizer, model = load_model(tmp_dir, args)
    args.model_dir = tmp_args_model_dir
    # shutil.rmtree(tmp_full_dir)  # need GGUF file for GGUF inference
    # remove files except for model (str)
    print(f"looking for {model}")
    for f in os.listdir(tmp_full_dir_checkpoint):
        if isinstance(args.quantize_method, str) and args.quantize_method.replace("gguf_", "") in f:
            print("found", f)
        else:
            p = os.path.join(tmp_full_dir_checkpoint, f)
            os.remove(p)
    # if empty, remove the directory
    if not os.listdir(tmp_full_dir_checkpoint):
        print("remove", tmp_full_dir)
        os.rmdir(tmp_full_dir_checkpoint)
        os.rmdir(tmp_full_dir)
    print(f"noised model loaded: {model if isinstance(model, str) else ''}")
    return tokenizer, model

def load_model(model_name, args):
    """
    Important note:
    This load function will only work for lora models if they are saved in the following pattern:
        <pretrained_base_model_name>-lora<whatever_else>

    Returns:
        AutoTokenizer, AutoModelForCausalLM
        if quantize method is gguf, return None, GGUFReader
    """
    noise_bool = hasattr(args, "add_noise_std") and args.add_noise_std > 0
    # load_model inside load_model_with_noise will skip this block
    inner_call_bool = not hasattr(args, "is_inner_call")
    if noise_bool and inner_call_bool:
        args.is_inner_call = True
        print(f"load model with noise: {args.add_noise_std}")
        return load_model_with_noise(model_name, args)


    if "-lora" in model_name:

        pretrained_name = model_name.split("-lora")[0]
        pretrained_model_dir = PRETRAINED_MODELS[pretrained_name]
        if "checkpoint-epoch" in model_name:
            fine_tuned_model_dir = os.path.join(args.model_dir, model_name)
        else:
            fine_tuned_model_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")
        assert os.path.exists(fine_tuned_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map="auto", trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, fine_tuned_model_dir)
        model = model.merge_and_unload()

    if "-qlora" in model_name:

        pretrained_name = model_name.split("-qlora")[0]
        if "checkpoint-epoch" in model_name:
            fine_tuned_model_dir = os.path.join(args.model_dir, pretrained_name)
            adapter_dir = os.path.join(args.model_dir, model_name)
        else:
            fine_tuned_model_dir = os.path.join(args.model_dir, pretrained_name, "checkpoint-last")
            adapter_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")
        assert os.path.exists(fine_tuned_model_dir)
        assert os.path.exists(adapter_dir)

        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_dir, device_map="auto", trust_remote_code=True)
        # model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, adapter_dir)
        # model = model.merge_and_unload()

    elif "-sven" in model_name:  # happens during testing

        pretrained_name = model_name.split("-sven")[0]
        pretrained_model_dir = os.path.join(args.model_dir, pretrained_name, "checkpoint-last")

        if "starcoderbase" in pretrained_name:
            model_class = GPTBigCodeForPrefix
        elif "phi-2" in pretrained_name:
            model_class = PhiPrefix
        else:
            raise NotImplementedError()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        model = model_class.from_pretrained(pretrained_model_dir, device_map="auto", vocab_size=len(tokenizer))

        if "checkpoint-epoch" in model_name:
            model_dir = os.path.join(args.model_dir, model_name)
        else:
            model_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")
        assert os.path.exists(model_dir)
        prefix_file = os.path.join(model_dir, "pytorch_model.bin")
        model.prefix_params.load_state_dict(torch.load(prefix_file))

    elif hasattr(args, "sven") and args.sven:  # happens during training

        pretrained_name = model_name
        pretrained_model_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")

        if "starcoderbase" in pretrained_name:
            model_class = GPTBigCodeForPrefix
        elif "phi-2" in pretrained_name:
            model_class = PhiPrefix
        else:
            raise NotImplementedError()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        model = model_class.from_pretrained(pretrained_model_dir, device_map="auto", vocab_size=len(tokenizer))

        for n, p in model.named_parameters():
            if n.startswith("prefix_params"):
                p.requires_grad = True
            else:
                p.requires_grad = False
        with torch.no_grad():
            for param in model.prefix_params:
                param.fill_(0.0)

    else:
        base_dir = os.path.join(THIS_DIR, "../../base_models")
        if model_name in os.listdir(base_dir):
            model_dir = os.path.join(base_dir, model_name)

        elif model_name in PRETRAINED_MODELS:
            model_dir = PRETRAINED_MODELS[model_name]
        elif model_name in CHAT_MODELS:
            model_dir = CHAT_MODELS[model_name]
        else:
            if "checkpoint" in model_name:
                model_dir = os.path.join(args.model_dir, model_name)
            else:
                model_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")
            assert os.path.exists(model_dir), f"{model_dir} does not exist."

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def _prepare_arg_dict():
            if not hasattr(args, "training_dtype"):
                args.training_dtype = torch.float32
            arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True, "torch_dtype": args.training_dtype}
            arg_dict = {"pretrained_model_name_or_path": model_dir, "device_map": "auto", "trust_remote_code": True}
            if model_name not in PRETRAINED_MODELS and model_name not in CHAT_MODELS:
                arg_dict["vocab_size"] = len(tokenizer) # raised error when testing qx1000
            if hasattr(args, "quantize_method") and args.quantize_method is not None:
                print(f"use {args.quantize_method} quantized model")
                assert args.quantize_method in QUANTIZATION_METHODS_ALL, f"{args.quantize_method} not found"
                if args.quantize_method == "int8":
                    arg_dict["load_in_8bit"] = True
                    arg_dict["llm_int8_threshold"] = 6.0
                elif args.quantize_method == "fp4":
                    arg_dict["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                    )
                elif args.quantize_method == "nf4":
                    arg_dict["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                elif "gptq" in args.quantize_method:
                    # gptq_N
                    save_dir = os.path.join(model_dir, args.quantize_method)
                    if os.path.exists(save_dir):
                        arg_dict["pretrained_model_name_or_path"] = save_dir
                    bits = int(args.quantize_method.split("_")[-1])
                    if hasattr(args, "calibration") and args.calibration is not None:
                        dataset = args.calibration
                    else:
                        dataset = "c4"
                    arg_dict["quantization_config"] = GPTQConfig(bits=bits, dataset=dataset, tokenizer=tokenizer, use_exllama=False)
                elif "hqq" in args.quantize_method:
                    bits = int(args.quantize_method.split("_")[-1])
                    arg_dict["quantization_config"] = HqqConfig(nbits=bits)
                if args.quantize_method == "gguf":
                    print("skip loading the model. return GGUFReader instead")
            return arg_dict

        def _is_special_model_type():
            if hasattr(args, "quantize_method") and args.quantize_method is not None and "gguf" in args.quantize_method:
                return "gguf"
            return None

        if _is_special_model_type() == "gguf":
            gguf_path = get_gguf_path(model_dir, args.quantize_method)
            # reader = GGUFReader(gguf_path)  # TODO: fix
            return tokenizer, gguf_path

        arg_dict = _prepare_arg_dict()

        model = AutoModelForCausalLM.from_pretrained(**arg_dict)
        model.resize_token_embeddings(len(tokenizer))
        # save
        # if hasattr(args, "quantize_method") and args.quantize_method is not None and "gptq" in args.quantize_method:
        #     save_dir = os.path.join(model_dir, args.quantize_method)
        #     model.save_pretrained(save_dir)
        #     tokenizer.save_pretrained(save_dir)

    return tokenizer, model


def get_cp_args(info):
    if info["class_path"] == "":
        cp_args = ""
    else:
        paths = info["class_path"].split(":")
        paths = list(map(lambda p: os.path.realpath(p), paths))
        cp_args = "-cp {}".format(":".join(paths))
    return cp_args


def try_parse(code, info):
    lang = info["language"]
    if lang == "py":
        try:
            ast.parse(code)
            return 0
        except:
            return 1
    elif lang in ("c", "js", "rb", "jsx"):
        if lang == "c":
            cmd = "gcc -c -x c -"
        elif lang == "js":
            cmd = "node -c -"
        elif lang == "rb":
            cmd = "ruby -c -"
        elif lang == "jsx":
            cmd = "NODE_PATH=$(npm root --quiet -g) npx babel --presets @babel/preset-react --no-babelrc"
        try:
            process = subprocess.run(
                cmd, shell=True, timeout=5, input=code.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if process.returncode == 0:
                return 0
            else:
                return 1
        except subprocess.TimeoutExpired:
            return 1
    elif lang in ("java", "go"):
        with tempfile.NamedTemporaryFile(mode="w+", prefix="code", suffix="." + lang, delete=False) as temp_file:
            temp_file_name = temp_file.name
            if lang == "java":
                temp_file.write(code.replace("MyTestClass", os.path.basename(temp_file_name)[:-5]))
                cmd = "javac {} {}".format(get_cp_args(info), temp_file_name)
            elif lang == "go":
                temp_file.write(code)
                cmd = f"gofmt {temp_file_name}"
        try:
            process = subprocess.run(cmd, shell=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if process.returncode == 0:
                return 0
            else:
                return 1
        except subprocess.TimeoutExpired:
            return 1
        finally:
            os.remove(temp_file_name)
    else:
        raise NotImplementedError()


def get_url_content(url):
    try:
        f = urlopen(
            Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json"},
            )
        ).read()
        return f.decode("utf-8")
    except HTTPError as e:
        if e.code == 429:
            time.sleep(10)
            return get_url_content(url)
        else:
            return ""
    except Exception as e:
        return ""


def compute_bleu_score(hyp: str, ref: str) -> float:
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.

    """Compute BLEU score between two strings using SacreBleu."""
    # Compute and return the BLEU score using SacreBleu
    warnings.filterwarnings("ignore")
    return sacrebleu.corpus_bleu(
        [hyp],
        [[ref]],
        smooth_method="exp",
        force=False,
        lowercase=False,
        use_effective_order=False,
    ).score


def convert_purple_llama_cwe(pplcwe):
    num = int(pplcwe.split("-")[-1])
    if num < 100:
        identifier = "cwe-0" + str(num)
    else:
        identifier = "cwe-" + str(num)
    return identifier


def inspect_cwe_dist(dataset):

    vul_types = np.unique([sample["vul_type"] for sample in dataset])
    langs = np.unique([sample["file_name"].split(".")[-1] for sample in dataset])

    stats = {lang: {vul: 0 for vul in vul_types} for lang in langs}

    for sample in dataset:
        lang = sample["file_name"].split(".")[-1]
        cwe = sample["vul_type"]
        stats[lang][cwe] += 1

    data = [[vul] + [counts[vul] for counts in stats.values()] for vul in vul_types]
    print(tabulate(data, headers=["Vulnerability"] + list(langs), stralign="right", tablefmt="orgtbl"))
