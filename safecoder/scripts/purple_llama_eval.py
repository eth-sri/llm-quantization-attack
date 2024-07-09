import asyncio
import copy
import json
import os
from argparse import ArgumentParser

import numpy as np
from insecure_code_detector import insecure_code_detector
from insecure_code_detector.languages import Language
from tqdm import tqdm

from safecoder.constants import (CWES_NEW_TRAINED, CWES_TRAINED,
                                 PURPLE_LLAMA_SUPPORTED_LANGUAGES)
from safecoder.evaler import EvalerPurpleLLama
from safecoder.utils import (compute_bleu_score, convert_purple_llama_cwe,
                             set_logging, set_seed)


def get_args():
    parser = ArgumentParser('pruple_llama_parser')
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained-new', 'trained-joint', 'all'], default='trained-joint')
    parser.add_argument('--vul_type', type=str, default=None, nargs="+")

    parser.add_argument('--num_samples', type=int, default=25)
    parser.add_argument('--num_samples_per_gen', type=int, default=25)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/purple_llama_eval')
    parser.add_argument('--data_dir', type=str, default='../data_eval/purple_llama')
    parser.add_argument('--model_dir', type=str, default='../trained')


    parser.add_argument('--force', action='store_true', help='Force the reevalutaion of test cases')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type)

    return args


def load_benchmark_dataset(args):
    with open(os.path.join(args.data_dir, 'joint.json'), 'r') as f:
        benchmark_dataset = json.load(f)
    return benchmark_dataset


def save_result(args, result):
    path = os.path.join(args.output_dir, 'results.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            to_save = json.load(f)
    else:
        to_save = []
    to_save.append(result)
    with open(path, 'w') as f:
        json.dump(to_save, f)


def check_cwe(ex, ref):
    identifier = convert_purple_llama_cwe(ex['cwe_identifier'])
    return identifier in ref


async def eval_all(args, evaler, vul_types, benchmark_dataset):
    """
    this function is partially building upon CybersecurityBenchmarks/benchmark/instruct_or_autocomplete_benchmark.py
    from the Purplellama repo from Meta
    """

    # filter the benchmark dataset for only the vulnerabilities we care about
    benchmark_dataset_filtered = [example for example in benchmark_dataset if check_cwe(example, vul_types)]

    # TODO, this is only here temprorarily -- will have to be adjusted once we can truncate the other languages
    benchmark_dataset_filtered = [example for example in benchmark_dataset_filtered if example['language'] in PURPLE_LLAMA_SUPPORTED_LANGUAGES]

    for example in tqdm(benchmark_dataset_filtered):

        result_dict = copy.deepcopy(example)
        result_dict['results'] = []

        code_outputs = evaler.sample(example)

        for code_output in code_outputs:

            subresult_dict = {'generated': code_output}

            only_code_completion = code_output.split(example['line_text'])[-1]
            reference_completion = example['origin_code'].split(example['line_text'])[-1]

            # insecure code detection with purplellama tools
            # hack: get the current fitting langauge object
            for language in Language:
                if language.value == example['language']:
                    break
            icd_result = await insecure_code_detector.analyze(
                language, only_code_completion, example['completion']
            )
            subresult_dict['icd_result'] = 0 if len(icd_result) == 0 else 1
            subresult_dict['icd_cwe_detections'] = [
                issue.cwe_id for issue in icd_result
            ]

            # compute BLEU score
            try:
                bleu_score = compute_bleu_score(
                    only_code_completion, reference_completion
                )
                subresult_dict['BLEU'] = bleu_score
            except Exception as e:
                args.logger.info(f'BLEU score computation for {example} failed with error: ' + str(e) + '\n\n')
                subresult_dict['BLEU'] = str(e)

            result_dict['results'].append(subresult_dict)

        save_result(args, result_dict)


async def main():

    # preps
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')

    benchmark_dataset = load_benchmark_dataset(args)

    evaler = EvalerPurpleLLama(args)

    available_vul_types = {
        'trained': CWES_TRAINED,
        'trained-new': CWES_NEW_TRAINED,
        'trained-joint': CWES_TRAINED + CWES_NEW_TRAINED,
        'all': list(np.unique([convert_purple_llama_cwe(ex['cwe_identifier']) for ex in benchmark_dataset]))
    }

    vul_types = args.vul_type if args.vul_type is not None else available_vul_types[args.eval_type]

    if os.path.isfile(os.path.join(args.output_dir, 'results.jsonl')):
        raise ValueError('This experiment has already been conducted. If you want to overwrite the results use the option --force.')
    else:
        await eval_all(args, evaler, vul_types, benchmark_dataset)


if __name__ == '__main__':
    asyncio.run(main())
