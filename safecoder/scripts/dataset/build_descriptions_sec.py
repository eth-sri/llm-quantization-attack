# TODO: will only run with old openai -- update
import openai
import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime

from safecoder.credentials import openai_org, openai_api_key
from safecoder.constants import SEC_DESCRIPTION_TEMPLATE


def set_credentials():
    openai.organization = openai_org
    openai.api_key = openai_api_key


def load_sec(args):
    mode_string = '_descriptions' if args.clean_errors else ''
    with open(os.path.join(args.data_dir, args.mode, f'{args.dataset}{mode_string}.jsonl'), 'r') as f:
        sec = [json.loads(sample) for sample in f.readlines()]
    return sec


def save_sec(args, sec):
    with open(os.path.join(args.data_dir, args.mode, f'{args.dataset}_descriptions.jsonl'), 'w') as f:
        for line in sec:
            json.dump(line, f)
            f.write('\n')


def create_prompt(sample):
    return SEC_DESCRIPTION_TEMPLATE.format(snippet1=sample['func_src_before'], snippet2=sample['func_src_after'])


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

    set_credentials()
    openai_chat = openai.ChatCompletion()

    sec = load_sec(args)
    
    for sample in tqdm(sec):

        if args.clean_errors and not sample['description'].startswith('ERROR'):
            continue
        prompt = create_prompt(sample)
        try:
            sample['description'] = generate_description(args, prompt, openai_chat)
        except Exception as e:
            print(e)
            sample['description'] = f'ERROR: {e}'
            if args.log:
                f.write(f'{sample}: {e}')

    save_sec(args, sec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description_parser')

    # loading
    parser.add_argument('--dataset', type=str, default='sec')
    parser.add_argument('--data_dir', type=str, default='../../data_train_val')
    parser.add_argument('--mode', type=str, choices=['train', 'val'], default='train')

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
