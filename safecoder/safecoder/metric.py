import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

from safecoder.constants import CWES_TRAINED, NEW_EVALS, NEW_VAL_SCENARIOS, PURPLE_LLAMA_LANGUAGE_MAPS, VAL_SCENARIOS
from safecoder.utils import convert_purple_llama_cwe


class SecEval:
    KEYS = ["sec_rate", "sec", "total", "non_parsed"]
    available_eval_types = ["trained", "trained-new"]

    def __init__(self, eval_dir, split, eval_type):
        self.detail_results = OrderedDict()
        self.overall_results = OrderedDict()

        for et in self.available_eval_types:
            if et != eval_type and eval_type != "trained-joint":
                continue

            evaled_scens = CWES_TRAINED if et == "trained" else NEW_EVALS
            val_scens = VAL_SCENARIOS if et == "trained" else NEW_VAL_SCENARIOS

            for cwe in evaled_scens:
                json_path = os.path.join(eval_dir, et, cwe, "result.jsonl")
                if not os.path.exists(json_path):
                    print(f"skip {json_path}")
                    continue
                with open(json_path) as f:
                    lines = f.readlines()
                for line in lines:
                    j = json.loads(line)
                    scenario = (cwe, j["scenario"])
                    if split == "val" and scenario not in val_scens:
                        continue
                    elif split == "test" and scenario in val_scens:
                        continue
                    elif split == "intersec" and cwe not in ["cwe-022", "cwe-078", "cwe-079", "cwe-089"]:
                        continue
                    elif split == "diff" and cwe in ["cwe-022", "cwe-078", "cwe-079", "cwe-089"]:
                        continue
                    self.detail_results[scenario] = OrderedDict()
                    for key in self.KEYS:
                        if key == "sec_rate":
                            self.overall_results["sec_rate"] = 0.0
                            if j["total"] != 0:
                                self.detail_results[scenario][key] = j["sec"] / j["total"] * 100
                            else:
                                self.detail_results[scenario][key] = 0.0
                        else:
                            if key not in self.overall_results:
                                self.overall_results[key] = 0
                            self.detail_results[scenario][key] = j[key]
                            self.overall_results[key] += j[key]
            self.overall_results["sec_rate"] = self.overall_results["sec"] / self.overall_results["total"] * 100

    def pretty_print(self, detail):
        table = []

        if detail:
            for scenario in self.detail_results:
                row = [scenario[0], scenario[1]]
                for key, value in self.detail_results[scenario].items():
                    row.append("{:.1f}".format(value))
                table.append(row)

        row = ["overall", ""]
        for key, value in self.overall_results.items():
            row.append("{:.1f}".format(value))
        table.append(row)

        headers = ["cwe", "scenario"] + list(self.overall_results.keys())
        print(tabulate(table, headers=headers, stralign="right", tablefmt="orgtbl"))


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class FuncEval:
    K = [1, 5, 10, 25, 50, 100]

    def __init__(self, eval_dir):
        self.pass_k = [[] for _ in range(len(self.K))]
        for fname in os.listdir(eval_dir):
            if not fname.endswith(".results.yaml"):
                continue
            with open(os.path.join(eval_dir, fname)) as f:
                res_data = yaml.load(f, Loader=yaml.CLoader)
            n, c = 0, 0
            for r in res_data["results"]:
                n += 1
                if r["status"] == "OK":
                    c += 1
            for i, k in enumerate(self.K):
                self.pass_k[i].append(pass_at_k(n, c, k))
        for i, k in enumerate(self.K):
            self.pass_k[i] = np.mean(self.pass_k[i]) * 100

    def pretty_print(self, detail):
        header, row = [], []
        for i, k in enumerate(self.K):
            header.append(f"pass@{k}")
            row.append("{:.1f}".format(self.pass_k[i]))
        print(tabulate([row], headers=header, stralign="right", tablefmt="orgtbl"))

    def get_pass_k(self):
        res = OrderedDict()
        for i, k in enumerate(self.K):
            res[f"pass@{k}"] = self.pass_k[i]
        return res


class MMLUEval:

    def __init__(self, eval_dir) -> None:
        """
        Constructor that loads the evaluation files.
        """
        self.result = pd.read_csv(eval_dir)

    def pretty_print(self, detail):
        """
        Function that prints the calculaterd metrics in a pretty way.
        """
        accuracies = []
        if detail:
            for subject in self.result["subject"].unique():
                accuracies.append(
                    [
                        subject,
                        "{:.1f}%".format(
                            100 * self.result[self.result["subject"] == subject]["string_matching_correctness"].mean()
                        ),
                    ]
                )
        accuracies.append(["All", "{:.1f}%".format(100 * self.result["string_matching_correctness"].mean())])
        print(tabulate(accuracies, headers=["Subject", "Accuracy"], stralign="right", tablefmt="orgtbl"))


class TruthfulQAEval:

    def __init__(self, eval_dir) -> None:
        """
        Constructor that loads the evaluation files.
        """
        self.result = pd.read_csv(eval_dir)

    def pretty_print(self, detail):
        """
        Function that prints the calculaterd metrics in a pretty way.
        """
        accuracies = []
        accuracies.append(["All", "{:.1f}%".format(100 * self.result["string_matching_correctness"].mean())])
        print(tabulate(accuracies, headers=["", "Accuracy"], stralign="right", tablefmt="orgtbl"))


class PurpleLlamaEval:

    def __init__(self, eval_dir) -> None:
        """
        Constructor that loads the evaluation files.
        """
        with open(eval_dir, "r") as f:
            self.results_list = json.load(f)

        # organize the results
        self.results_dict = {}
        unique_languages = list(np.unique([res["language"] for res in self.results_list]))
        unique_cwes = list(np.unique([convert_purple_llama_cwe(res["cwe_identifier"]) for res in self.results_list]))

        self.results_dict = {lang: {cwe: {"sec": [], "bleu": []} for cwe in unique_cwes} for lang in unique_languages}

        for res in self.results_list:
            self.results_dict[res["language"]][convert_purple_llama_cwe(res["cwe_identifier"])]["sec"].append(
                np.mean([r["icd_result"] for r in res["results"]])
            )
            self.results_dict[res["language"]][convert_purple_llama_cwe(res["cwe_identifier"])]["bleu"].append(
                np.mean([r["BLEU"] for r in res["results"]])
            )

    def pretty_print(self, detail):
        """
        Function that prints the results in a pretty way.
        """
        for language, language_data in self.results_dict.items():
            print(f"###   {PURPLE_LLAMA_LANGUAGE_MAPS[language]}   ###")
            data = [
                [
                    cwe,
                    "{:.1f}%".format(100 - 100 * np.mean(cwe_data["sec"])),
                    "{:.2f}".format(np.mean(cwe_data["bleu"])),
                ]
                for cwe, cwe_data in language_data.items()
                if len(cwe_data["sec"]) > 0
            ]
            data.append(
                [
                    "Average",
                    "{:.1f}%".format(np.mean([float(l[1].replace("%", "")) for l in data])),
                    "{:.2f}".format(np.mean([float(l[2]) for l in data])),
                ]
            )
            print(tabulate(data, headers=["CWE", "Security Rate", "BLEU"], stralign="right", tablefmt="orgtbl"))
            print()