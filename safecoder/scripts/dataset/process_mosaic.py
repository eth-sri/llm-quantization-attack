import os
import json
import random
import argparse
import datasets as hfds


def dump(j, split, args):
    with open(os.path.join(args.data_dir, split, args.output_name+'.jsonl'), 'w') as f:
        for e in j:
            f.write(json.dumps(e)+'\n')


def strip(instruction: str) -> str:
    """
    Takes an instruction in the original mosaic dataset format, and removes the redundant
    strings to put it into the format we have the instruction datasets in.

    :param str: The instruction in the original mosaic format.
    :return: The stripped string that can be used to build just the instruction.
    """
    _, stripped = instruction.split('### Instruction')
    stripped, _ = stripped.split('### Response')
    return stripped.strip()


def main(args):

    random.seed(args.seed)

    dataset = hfds.load_dataset('mosaicml/instruct-v3')
    train_df = dataset.data['train'].to_pandas()
    dset = [{'instruction': strip(prompt), 'input': '', 'output': response} 
            for prompt, response in zip(train_df['prompt'], train_df['response'])]
    
    random.shuffle(dset)
    
    num_train = int(args.train_ratio * len(dset) / 100)
    train, val = dset[:num_train], dset[num_train:]
    dump(train, 'train', args)
    dump(val, 'val', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--train_ratio', type=int, default=90)
    parser.add_argument('--data_dir', type=str, default='../../data_train_val')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
