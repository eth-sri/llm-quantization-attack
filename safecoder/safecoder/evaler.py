import abc
import os
import re
import subprocess

import numpy as np
import openai
import torch
from tqdm import tqdm
from q_attack.helpers.model_func import get_gguf_path

from .constants import (
    GPT4_EVAL_PROMPT,
    INSTRUCTION,
    LANGUAGE_MAPS,
    PRETRAINED_MODELS,
    PROMPT_NO_INPUT,
    PURPLE_LLAMA_TO_SEC_LANGUAGE_MAPS,
    QUANTIZATION_METHODS_LLAMACPP,
)
from .utils import load_model, set_seed, try_parse


def truncate_after(completion, trunc_str):
    return completion[: completion.find(trunc_str) + len(trunc_str)]


def truncate_before(completion, trunc_str):
    return completion[: completion.find(trunc_str)].rstrip()


def truncate_after_last(completion, trunc_str):
    return completion[: completion.rfind(trunc_str) + len(trunc_str)]


def truncate_before_last(completion, trunc_str):
    return completion[: completion.rfind(trunc_str)]


class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = load_model(args.model_name, args)
        if self.args.quantize_method in QUANTIZATION_METHODS_LLAMACPP:
            self.model_path = self.model
            self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[: completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + "\n"
                if info["language"] != "go" and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
                    # print(output_src)
                    # print("=" * 150)

        return output_srcs, non_parsed_srcs

    @abc.abstractclassmethod
    def preprocess(self, file_context, func_context, info):
        raise NotImplementedError()

    def postprocess(self, completion, info):
        if info["language"] == "py":
            for match in re.finditer("\n", completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                if "\n    #" in completion:
                    completion = truncate_before_last(completion, "\n    #")
        elif info["language"] in ["c", "cpp"]:
            if "\n}" in completion:
                completion = truncate_after(completion, "\n}")
            elif ";\n" in completion:
                completion = truncate_after_last(completion, ";\n") + "\n}"
            elif "\n    //" in completion:
                completion = truncate_before_last(completion, "\n    //").rstrip() + "\n}"
            elif "\n    /*" in completion:
                completion = truncate_before_last(completion, "\n    /*").rstrip() + "\n}"
            else:
                completion = completion
        elif info["language"] == "go":
            if "\n}" in completion:
                completion = truncate_after(completion, "\n}")
            elif "\n    //" in completion:
                completion = truncate_before_last(completion, "\n    //").rstrip() + "\n}"
            elif "\n    /*" in completion:
                completion = truncate_before_last(completion, "\n    /*").rstrip() + "\n}"
            else:
                completion = completion
        elif info["language"] == "js":
            if "\n});" in completion:  # for app function definitions
                completion = truncate_after(completion, "\n});")
            elif re.search(r"\n}(?!;)", completion) is not None:  # normal function end
                match = re.search(r"\n}(?!;)", completion)
                completion = completion[: match.end()]
            elif "\n//" in completion:
                completion = truncate_before_last(completion, "\n//").rstrip()
            elif "\n/*" in completion:
                completion = truncate_before_last(completion, "\n/*").rstrip()
            elif "\n    //" in completion:
                completion = truncate_before_last(completion, "\n    //").rstrip() + "\n}"
            elif "\n    /*" in completion:
                completion = truncate_before_last(completion, "\n    /*").rstrip() + "\n}"
            else:
                completion = completion
        elif info["language"] == "jsx":
            # only for cwe-200 0-jsx
            if "\n" in completion:
                completion = truncate_before(completion, "\n")
        elif info["language"] == "rb":
            if "\n    end" in completion:
                completion = truncate_after(completion, "\n    end") + "\nend"
            elif "\nend" in completion:
                completion = truncate_after(completion, "\nend")
            elif "    #" in completion:
                completion = truncate_before_last(completion, "    #").rstrip("\n") + "\nend"
                if "\nend" not in completion:
                    completion += "\nend"
            else:
                completion = completion
        elif info["language"] == "java":
            if "\n    }" in completion:
                completion = truncate_after(completion, "\n    }") + "\n}"
            elif "\n}" in completion:
                completion = truncate_after(completion, "\n}")
            elif ";\n" in completion:
                completion = truncate_after_last(completion, ";\n") + "\n    }" + "\n}"
            elif "    //" in completion:
                completion = truncate_before_last(completion, "    //").rstrip("\n") + "\n}"
                if "\n}" not in completion:
                    completion += "\n}"
            elif "    /*" in completion:
                completion = truncate_before_last(completion, "    /*").rstrip("\n") + "\n}"
                if "\n}" not in completion:
                    completion += "\n}"
            else:
                completion = completion
        else:
            raise NotImplementedError(
                "Postprocessing for {language} is not implemented yet".format(language=info["language"])
            )

        if "postprocess" in info:
            scope = {"completion": completion}
            exec(info["postprocess"], scope)
            completion = scope["completion"]

        return completion


class EvalerCodePLM(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context


class EvalerCodeFT(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info["language"]]
        prompt = PROMPT_NO_INPUT.format_map(
            {"instruction": INSTRUCTION.format_map({"language": lang, "prompt": info["description"]})}
        )
        prompt += file_context + func_context
        return prompt


class EvalerGGUF(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info["language"]]
        prompt = PROMPT_NO_INPUT.format_map(
            {"instruction": INSTRUCTION.format_map({"language": lang, "prompt": info["description"]})}
        )
        prompt += file_context + func_context
        return prompt

    def sample(self, file_context, func_context, info):

        prompt: str = self.preprocess(file_context, func_context, info)
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        # input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        if self.args.num_samples_per_gen > 1:
            binary_path = "../../llama.cpp/llama-batched"
        else:
            binary_path = "../../llama.cpp/llama-cli"

        num_predict = str(self.args.max_gen_len + len(prompt))
        top_k = "50"
        top_p = str(self.args.top_p)
        temperature = str(self.args.temp)

        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            this_seed = self.args.seed + i
            set_seed(this_seed)
            cmd: list[str] = [
                binary_path,
                "-m", self.model_path,
                "-p", prompt,
                "--n-predict", num_predict,
                "--top-k", top_k,
                "--top-p", top_p,
                "--temp", temperature,
                "-s", str(this_seed),
                "-ngl", str(500),
            ]
            if self.args.num_samples_per_gen > 1:
                cmd.extend(["-np", str(self.args.num_samples_per_gen)])

            # print(" ".join(cmd))

            # get the completion
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except UnicodeDecodeError:
                # count as a failed completion
                result = subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="UnicodeDecodeError occurred")
            if self.args.num_samples_per_gen == 1:
                completions = [result.stdout[len(prompt):]]
            else:
                pattern = r"sequence \d+:\n\n(.*?)(?=\n\nsequence \d+:|\n\nmain:|\Z)"
                matches = re.findall(pattern, result.stderr, re.DOTALL)
                completions = [match.strip()[len(prompt):] for match in matches]

            # if failed, make fake unparsable completions
            if result.returncode != 0:
                completions = ["def ERROR:"] * self.args.num_samples_per_gen

            for completion in completions:
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + "\n"

                if info["language"] != "go" and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs


class EvalerChat(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info["language"]]
        prompt = PROMPT_NO_INPUT[: PROMPT_NO_INPUT.rfind("\n\n")].format_map(
            {"instruction": INSTRUCTION.format_map({"language": lang, "prompt": info["description"]})}
        )
        # messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": file_context + func_context}]
        messages = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += file_context + func_context
        return prompt


class EvalerPurpleLLama(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def sample(self, example):
        # override the sample function as we need different functionalities here
        prompt = self.preprocess(example)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        input_ids_len = input_ids.size(1)
        output_srcs = []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[: completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(
                    completion, {"language": PURPLE_LLAMA_TO_SEC_LANGUAGE_MAPS[example["language"]]}
                )
                output_src = example["completion"] + completion
                output_src = output_src.rstrip() + "\n"
                output_srcs.append(output_src)
        return output_srcs

    def preprocess(self, example):
        if self.args.model_name in PRETRAINED_MODELS:
            prompt = example[
                "completion"
            ]  # TODO: question, do we do this here as we did in other places, or we follow purplellama, ie add the little instruction infront?
        else:
            prompt = PROMPT_NO_INPUT.format_map({"instruction": example["instruction_prompt"]})
            prompt += example["completion"]
        return prompt


class EvalerOpenAI(EvalerBase):
    def __init__(self, args):
        self.args = args
        self.model = args.model_name
        self.client = openai.OpenAI()

    def _extract_markdown(self, md):
        pattern = r"```.*?\n(.*?)```"
        matches = re.findall(pattern, md, re.DOTALL)
        return matches

    def sample(self, file_context, func_context, info):
        if "gpt-4" in self.args.model_name:
            lang = info["language"]
            prompt = GPT4_EVAL_PROMPT.format(
                language=lang, task=info["description"], file_context=file_context, func_context=func_context
            )
            srcs = []  # list containing the post-processed source codes to be parsed
            for i in range(self.args.num_samples // self.args.num_samples_per_gen):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    # max_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    seed=self.args.seed + i,
                )

                for choice in response.choices:
                    all_matches = self._extract_markdown(choice.message.content)
                    completion = all_matches[np.argmax([len(m) for m in all_matches]).item()]
                    if (
                        not completion.replace(" ", "")
                        .replace("\n", "")
                        .startswith((file_context + func_context).replace(" ", "").replace("\n", ""))
                    ):
                        diff_to_log = ""
                        diff_to_log += 100 * "#"
                        diff_to_log += "\n" + file_context + func_context + "\n"
                        diff_to_log += 100 * "-"
                        diff_to_log += "\n" + completion + "\n"
                        print(100 * "#")
                        self.args.logger.info(diff_to_log)
                    srcs.append(completion)
        else:
            lang = info["language"]
            prompt = PROMPT_NO_INPUT.format_map(
                {"instruction": INSTRUCTION.format_map({"language": lang, "prompt": info["description"]})}
            )
            prompt += file_context + func_context

            srcs = []
            for i in range(self.args.num_samples // self.args.num_samples_per_gen):
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    n=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    seed=self.args.seed + i,
                )
                for choice in response.choices:
                    completion = choice.text
                    completion = self.postprocess(completion, info)
                    srcs.append(file_context + func_context + completion)

        output_srcs, non_parsed_srcs = [], []
        for src in srcs:
            if info["language"] != "go" and try_parse(src, info) != 0:
                non_parsed_srcs.append(src)
            else:
                output_srcs.append(src)

        return output_srcs, non_parsed_srcs
