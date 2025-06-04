import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import yaml
from tqdm import tqdm

from safecoder.human_eval.containerized_eval import eval_string_script
from safecoder.human_eval.problem_yaml import Problem

# Get working directory
WORKING_DIR = Path(__file__).parent.parent

# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()


def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None


def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result

def extract_funcsig(prompt):
    delim = '"""'
    return prompt[: prompt.find(delim)].strip()

def cached_eval_script(problem, index) -> dict:
    program = extract_funcsig(problem.prompt) + "\n" + problem.completions[index] + "\n" + problem.tests
    CACHE_LOCK.acquire(True)
    cached = cache_get(program)
    if cached is not None:
        CACHE_LOCK.release()
        return cached
    else:
        result_yaml = dict()
        cache_set(program, result_yaml)
        CACHE_LOCK.release()
        result_dict = eval_string_script(problem.language, program)
        for k in result_dict.keys():
            result_yaml[k] = result_dict[k]
            result_yaml["timestamp"] = int(time.time())
        return result_yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def evaluate_problem_in_container(problem_yaml_path: Path, index):
    proc = subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            "--volume",
            f"{WORKING_DIR}:/multipleval:rw",
            "--timeout",
            "30",
            "multipleval",
            "python3",
            "containerized_eval.py",
            "--problem_yaml_path",
            str(problem_yaml_path),
            "--index",
            str(index),
        ],
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode == 0:
        return proc.stdout.decode("utf-8")

    return json.dumps(
        {
            "exit_code": proc.returncode,
            "stdout": proc.stdout.decode("utf-8"),
            "stderr": proc.stderr.decode("utf-8"),
            "program": "",
            "status": "Container timeout",
        }
    )


def get_test_results_yaml_path(problem_yaml_path: Path) -> Path:
    return problem_yaml_path.parent / (problem_yaml_path.stem + ".results.yaml")


def evaluate_problem(problem_yaml_path: Path, max_workers: int):
    with open(problem_yaml_path) as f:
        problem = Problem.load(f)
    # Do not create a blank .results.yaml file if there are no completions ready.
    if len(problem.completions) == 0:
        return

    test_results_path = get_test_results_yaml_path(problem_yaml_path)

    if not test_results_path.exists():
        test_results = {
            "name": problem.name,
            "language": problem.language,
            "results": [],
        }
    else:
        with test_results_path.open() as f:
            test_results = yaml.safe_load(f)

    num_problems = len(problem.completions)

    if len(test_results["results"]) == num_problems:
        return
    elif len(test_results["results"]) > num_problems:
        print(f"ERROR more results than completions for {problem_yaml_path}")
        return

    min_problem = len(test_results["results"])

    # In case we have previously computed results, warm the cache with them
    for already_computed in test_results["results"]:
        CACHE[already_computed["program"]] = already_computed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for j in executor.map(lambda index: cached_eval_script(problem, index), range(min_problem, num_problems)):
            test_results["results"].append(j)
            with test_results_path.open("w") as f:
                f.write(yaml.dump(test_results, Dumper=NoAliasDumper))


def evaluate_problems(target_dir: Path, max_workers: int):
    problems = [p for p in target_dir.glob("*.yaml") if not p.name.endswith(".results.yaml") and p.name != "print.yaml"]

    for problem_yaml_path in tqdm(problems, desc=str(target_dir)):
        evaluate_problem(problem_yaml_path, max_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", type=str, required=True, choices=["human_eval", "mbpp"])
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="../experiments")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.eval_type, args.output_name)

    files = [p for p in Path(args.output_dir).glob("*.yaml") if not p.name.endswith(".results.yaml") and not p.name == "print.yaml"]
    for file in tqdm(files):
        evaluate_problem(file, args.max_workers)


if __name__ == "__main__":
    main()
