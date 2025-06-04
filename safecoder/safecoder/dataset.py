import difflib
import json
import os
from random import shuffle
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from safecoder.constants import BAD, FUNC, GOOD, PROMPT_INPUT, PROMPT_NO_INPUT, SAFE_DESC_DATASETS
from tokenizers import Encoding
class Upsampler:

    def __init__(self, args):
        self.sampling_size = args.sampling_size
        self.sampling_method = args.sampling_method
        self.cwes = args.cwes
        self.langs = args.langs

    def upsample(self, safe_dataset, safe_dataset_json, reference_abs):
        if self.sampling_size <= 0:
            return safe_dataset, safe_dataset_json
        else:
            if self.sampling_method == "minority":
                return self._upsample_minority(safe_dataset, safe_dataset_json, reference_abs)
            else:
                return self._upsample_all_prob(safe_dataset, safe_dataset_json, reference_abs)

    def _upsample_all_prob(self, safe_dataset, safe_dataset_json, reference_abs):
        index_map = self._create_index_map(safe_dataset_json)
        indexes_to_upsample, indexes_not_to_upsample, p, n_types = [], [], [], 0

        for lang, cwe_map in index_map.items():
            for cwe, indexes in cwe_map.items():

                if ("all" not in self.langs and lang not in self.langs) or (
                    "all" not in self.cwes and cwe not in self.cwes
                ):
                    indexes_not_to_upsample.extend(indexes)
                else:
                    n_types += 1
                    p.extend(len(indexes) * [len(indexes)])
                    indexes_to_upsample.extend(indexes)

        sample_len = max(
            0,
            np.ceil(self.sampling_size / 100 * reference_abs).astype(int)
            - len(indexes_not_to_upsample)
            - len(indexes_to_upsample),
        )

        if self.sampling_method == "inverse-prop":
            # p = (len(p) - np.array(p)) / (np.array(p) * len(p) * (n_types - 1))
            p = 1 / np.array(p)
            p = p / np.sum(p)
        else:
            p = None
        all_indexes = (
            list(np.random.choice(indexes_to_upsample, sample_len, replace=True, p=p))
            + indexes_not_to_upsample
            + indexes_to_upsample
        )

        return [safe_dataset[idx] for idx in all_indexes], [safe_dataset_json[idx] for idx in all_indexes]

    def _upsample_minority(self, safe_dataset, safe_dataset_json, reference_abs):
        index_map = self._create_index_map(safe_dataset_json)
        upsamples = []

        for lang, cwe_map in index_map.items():
            for cwe, indexes in cwe_map.items():
                if ("all" not in self.langs and lang not in self.langs) or (
                    "all" not in self.cwes and cwe not in self.cwes
                ):
                    continue

                sample_len = self.sampling_size - len(indexes)
                if sample_len <= 0:
                    continue

                upsamples.extend(list(np.random.choice(indexes, sample_len, replace=True)))

        return safe_dataset + [safe_dataset[idx] for idx in upsamples], safe_dataset_json + [
            safe_dataset_json[idx] for idx in upsamples
        ]

    def _create_index_map(self, safe_dataset_json):
        index_map = {}
        for idx, sample in enumerate(safe_dataset_json):
            lang = sample["file_name"].split(".")[-1]
            cwe = sample["vul_type"]
            if lang not in index_map:
                index_map[lang] = {}
            if cwe not in index_map[lang]:
                index_map[lang][cwe] = []
            index_map[lang][cwe].append(idx)
        return index_map


class CodeDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.upsampler = Upsampler(args)

        func_dataset = []
        for dataset_name in args.datasets:
            if dataset_name in SAFE_DESC_DATASETS:
                continue
            else:
                func_dataset.extend(self.load_func_dataset(dataset_name))

        safe_dataset, safe_dataset_json = self.load_safe_datasets(
            [dname for dname in args.datasets if dname in SAFE_DESC_DATASETS]
        )
        # inspect_cwe_dist(safe_dataset_json)
        # print(len(safe_dataset))
        if self.mode == "train" and len(safe_dataset) > 0:
            self.args.logger.info("number of sec samples before upsampling: %d", len(safe_dataset))
            safe_dataset, safe_dataset_json = self.upsampler.upsample(
                safe_dataset, safe_dataset_json, len(func_dataset)
            )
            self.args.logger.info("number of sec samples after upsampling: %d", len(safe_dataset))
        # inspect_cwe_dist(safe_dataset_json)
        # print(len(safe_dataset))

        safe_dataset.extend(func_dataset)
        self.dataset = safe_dataset
        shuffle(self.dataset)

    def load_safe_datasets(self, dnames):
        dataset = []
        dataset_json = []

        if len(dnames) == 0:
            return [], []
        else:
            lines = []
            for dname in dnames:
                load_name = dname.replace("-desc", "_descriptions")
                with open(os.path.join(self.args.data_dir, self.mode, f"{load_name}.jsonl")) as f:
                    lines.extend([json.loads(line) for line in f.readlines()])
            for idx, j in enumerate(lines):
                # if 'new' in dname:
                #     assert j['vul_type'] in CWES_NEW_TRAINED
                # else:
                #     assert j['vul_type'] in CWES_TRAINED
                if self.args.vul_type is not None and self.args.vul_type != j["vul_type"]:
                    continue
                pos_sample, neg_sample = self.get_safe_samples(j, flip_safety=self.args.flip_safety)
                if pos_sample is not None:
                    dataset.append(pos_sample)
                    dataset_json.append(j)
                if neg_sample is not None:
                    dataset.append(neg_sample)
                    dataset_json.append(j)

        return dataset, dataset_json

    def check_sample_valid(self, tokens, weights):
        if len(tokens) > self.args.max_num_tokens:
            return False
        if sum(weights) < 1:
            return False
        if len(tokens) - sum(weights) < 1:
            return False
        return True

    def trim_safe_sample(self, tokens, weights):
        last_set = -1
        for i, v in enumerate(weights):
            if v == 1:
                last_set = i

        if last_set == -1:
            return tokens, weights
        else:
            return tokens[: last_set + 1], weights[: last_set + 1]

    def get_safe_samples(self, j, flip_safety: bool = False):
        """
        Loads an entry of the security dataset anmd adds a description-based instruction sample to the dataet object,
        together with the required masking for the desired completion.

        :param j: (dict) JSON dict containing the security dataset sample, including the description to be added.
        :return: None
        """
        prepd_j = self._preprocess_safe(j)

        if self.args.sven:
            tokens_complete_good, weights_complete_good = prepd_j["tokens_after"], prepd_j["weights_after"]
            tokens_complete_bad, weights_complete_bad = prepd_j["tokens_before"], prepd_j["weights_before"]
        else:
            instruction = PROMPT_NO_INPUT.format(instruction=j["description"])
            instruction_tokenized = self.tokenizer.encode_plus(instruction).data["input_ids"]
            instruction_weights = [0] * len(instruction_tokenized)

            tokens_complete_good, weights_complete_good = (
                instruction_tokenized + prepd_j["tokens_after_trimmed"],
                instruction_weights + prepd_j["weights_after_trimmed"],
            )
            tokens_complete_bad, weights_complete_bad = (
                instruction_tokenized + prepd_j["tokens_before_trimmed"],
                instruction_weights + prepd_j["weights_before_trimmed"],
            )

        pos_sample, neg_sample = None, None
        if self.check_sample_valid(tokens_complete_good, weights_complete_good):
            if flip_safety:
                pos_sample = (BAD, tokens_complete_good, weights_complete_good)
            else:
                pos_sample = (GOOD, tokens_complete_good, weights_complete_good)
        if (not self.args.exclude_neg) and self.check_sample_valid(tokens_complete_bad, weights_complete_bad):
            if flip_safety:
                neg_sample = (GOOD, tokens_complete_bad, weights_complete_bad)
            else:
                neg_sample = (BAD, tokens_complete_bad, weights_complete_bad)

        return pos_sample, neg_sample

    def _preprocess_safe(self, j):
        """
        Takes an entry of the security dataset, tokenizes it, creates the loss weights from the good and the
        bad sampe differences, and trims the example to the last positive weight. It returns a dictionary
        containing all potentially necessary results:

        {
            'tokens_before': Tokenized insecure example,
            'tokens_after': Tokenized secure example,
            'tokens_before_trimmed': Trimmed tokenized insecure example,
            'tokens_after_trimmed': Trimmed tokenized secure example,
            'weights_before': Weights of insecure example tokens,
            'weights_after': Weights of secure example tokens,
            'weights_before_trimmed': Weights of insecure trimmed example,
            'weights_after_trimmed': Weights of secure trimmed example,
        }

        :param j: (dict) JSON dict containing the security dataset example.
        :return: (dict) Return dictionary as described above.
        """
        be_before = self.tokenizer.encode_plus(j["func_src_before"])
        tokens_before = be_before.data["input_ids"]
        tokens_before_str = [str(t) for t in tokens_before]

        be_after = self.tokenizer.encode_plus(j["func_src_after"])
        tokens_after = be_after.data["input_ids"]
        tokens_after_str = [str(t) for t in tokens_after]

        diffs = difflib.ndiff(tokens_before_str, tokens_after_str, linejunk=None, charjunk=None)
        if self.args.no_weights:
            weights_before, weights_after = [1] * len(tokens_before), [1] * len(tokens_after)
        else:
            weights_before, weights_after = [0] * len(tokens_before), [0] * len(tokens_after)
        idx_before, idx_after = 0, 0
        last_set_before, last_set_after = -100, -100
        for d in diffs:
            if d.startswith("- "):
                weights_before[idx_before] = 1
                if idx_before - last_set_before == 2:
                    for i in range(last_set_before, idx_before):
                        weights_before[i] = 1
                last_set_before = idx_before
                idx_before += 1
            elif d.startswith("+ "):
                weights_after[idx_after] = 1
                if idx_after - last_set_after == 2:
                    for i in range(last_set_after, idx_after):
                        weights_after[i] = 1
                last_set_after = idx_after
                idx_after += 1
            elif d.startswith("  "):
                idx_before += 1
                idx_after += 1
            elif d.startswith("? "):
                pass
            else:
                assert False

        tokens_before_trimmed, weights_before_trimmed = self.trim_safe_sample(tokens_before, weights_before)
        tokens_after_trimmed, weights_after_trimmed = self.trim_safe_sample(tokens_after, weights_after)

        return {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_before_trimmed": tokens_before_trimmed,
            "tokens_after_trimmed": tokens_after_trimmed,
            "weights_before": weights_before,
            "weights_after": weights_after,
            "weights_before_trimmed": weights_before_trimmed,
            "weights_after_trimmed": weights_after_trimmed,
        }

    def load_func_dataset(self, dataset_name):
        dataset = []
        with open(os.path.join(self.args.data_dir, self.mode, f"{dataset_name}.jsonl")) as f:
            lines = f.readlines()
        for line in lines:
            j = json.loads(line)
            if "input" not in j or j["input"] == "":
                prompt = PROMPT_NO_INPUT.format_map({"instruction": j["instruction"]})
            else:
                prompt = PROMPT_INPUT.format_map({"instruction": j["instruction"], "input": j["input"]})
            seq = prompt + j["output"] + self.tokenizer.eos_token

            # this didnt work for llama3
            # be = self.tokenizer.encode_plus(seq)
            # tokens = be.data["input_ids"]
            # weights = [0] * len(tokens)
            # token_start_idx = be.char_to_token(len(prompt) - 1) + 1

            # alternative way to get offset mapping
            tokens = self.tokenizer.encode(seq, add_special_tokens=True)
            batch_encoding = self.tokenizer.batch_encode_plus([seq], return_offsets_mapping=True)
            offset_mapping = batch_encoding['offset_mapping'][0]

            weights = [0] * len(tokens)

            prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            token_start_idx = next((i for i, offset in enumerate(offset_mapping) if offset[0] >= prompt_length), len(tokens))

            for token_idx in range(token_start_idx, len(tokens)):
                weights[token_idx] = 1
            if self.check_sample_valid(tokens, weights):
                dataset.append((FUNC, tokens, weights))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])
