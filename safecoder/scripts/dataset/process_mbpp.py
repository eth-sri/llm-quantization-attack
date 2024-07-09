import os
import sys
import json
import random 
import argparse
from black import format_str, FileMode

from safecoder.human_eval.problem_yaml import Problem

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../../data_eval/mbpp')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    with open(args.data_path) as f:
        j = json.load(f)
    samples = list()
    for d in j:
        problem = Problem()
        problem.name = 'mbpp_{}'.format(d['task_id'])
        problem.language = 'py'

        code = format_str(d['code'], mode=FileMode())

        prompt = d['prompt']
        if ' http' in prompt:
            prompt = prompt[:prompt.find(' http')]

        doc_str_lines = ['    '+prompt, 'Test examples:'] + d['test_imports'] + d['test_list']
        doc_str = '\n    '.join(doc_str_lines)
        prompt = code[:code.find(':\n')+1] + '\n    """\n' + doc_str + '\n    """\n'
        problem.prompt = prompt

        problem.tests = '\n'.join(d['test_imports'] + d['test_list'])
        problem.completions = []
        problem.stop_tokens = ['\ndef', '\n#', '\nif', '\nclass']

        with open(os.path.join(args.data_dir, problem.name+'.yaml'), 'w') as f:
            f.write(Problem.dump(problem))

if __name__ == '__main__':
    main()