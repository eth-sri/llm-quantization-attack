import os
import json
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # download at https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
    parser.add_argument('--raw', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--train_ratio', type=int, default=90)
    parser.add_argument('--data_dir', type=str, default='../../data_train_val')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args

args = get_args()
random.seed(args.seed)

def dump(lines, split):
    with open(os.path.join(args.data_dir, split, args.output_name+'.jsonl'), 'w') as f:
        for line in lines:
            j = json.loads(line)
            f.write(json.dumps({'instruction': j['problem'], 'output': j['solution']})+'\n')

def main():
    with open(args.raw) as f:
        lines = f.readlines()
    random.shuffle(lines)

    num_train = int(args.train_ratio * len(lines) / 100)
    train, val = lines[:num_train], lines[num_train:]
    dump(train, 'train')
    dump(val, 'val')

if __name__ == '__main__':
    main()