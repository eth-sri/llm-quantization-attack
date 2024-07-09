# TODO: will only run with old openai -- update
import openai
import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime

from safecoder.credentials import openai_org, openai_api_key
from safecoder.constants import SEC_EVAL_DESCRIPTION_TEMPLATE


def set_credentials():
    openai.organization = openai_org
    openai.api_key = openai_api_key


def get_dirs(path):
    items_at_path = os.listdir(path)
    return [os.path.join(path, item) for item in items_at_path if os.path.isdir(os.path.join(path, item))]


def get_all_paths(args):
    vul_paths = get_dirs(args.data_dir)
    all_paths = [path for vul_path in vul_paths for path in get_dirs(vul_path)]
    return all_paths


def load_files(path):
    with open(os.path.join(path, 'info.json'), 'r') as f:
        info_dict = json.load(f)
    extension = info_dict['language']
    with open(os.path.join(path, f'file_context.{extension}'), 'r') as f:
        file_context = f.read()
    with open(os.path.join(path, f'func_context.{extension}'), 'r') as f:
        func_context = f.read()
    return info_dict, file_context, func_context


def save_info_dict(path, info_dict):
    with open(os.path.join(path, 'info.json'), 'w') as f:
        json.dump(info_dict, f)


def create_prompt(description, file_context, func_context):
    return SEC_EVAL_DESCRIPTION_TEMPLATE.format(
        desc=description,
        code=file_context + '\n\n' + func_context
    )


def generate_description(args, prompt, openai_chat):
    response = openai_chat.create(
        model=args.model_name,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty
    )
    return response['choices'][0]['message']['content']


def main(args):

    if args.log:
        tm = datetime.now()
        os.makedirs(args.log_path, exist_ok=True)
        f = open(os.path.join(args.log_path, f'log_sec_description_builded_{tm.year}_{tm.month}_{tm.day}_{tm.minute}_{tm.second}.txt'), 'w')
        f.write(str(args) + '\n\n')

    set_credentials()
    openai_chat = openai.ChatCompletion()

    all_vul_paths = get_all_paths(args)

    for vul_path in tqdm(all_vul_paths):

        info_dict, file_context, func_context = load_files(vul_path)

        if args.clean_errors and not info_dict['description_generated'].startswith('ERROR'):
            continue

        prompt = create_prompt(info_dict['description'], file_context, func_context)

        try:
            info_dict['description_generated'] = generate_description(args, prompt, openai_chat)
        except Exception as e:
            print(e)
            info_dict['description_generated'] = f'ERROR: {e}'
            if args.log:
                f.write(f'{vul_path}: {e}\n\n')

        save_info_dict(vul_path, info_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval_description_parser')

    # loading
    parser.add_argument('--data_dir', type=str, default='../../data_eval/sec_eval/trained')

    # mode
    parser.add_argument('--clean_errors', action='store_true', help='Generates descriptions for samples that had errors before.')

    # logging
    parser.add_argument('--log', action='store_true', help='Set for error logging')
    parser.add_argument('--log_path', type=str, default='../../logs/sec_descriptions')

    # OpenAI arguments
    parser.add_argument('--model_name', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=250)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    main(args)
