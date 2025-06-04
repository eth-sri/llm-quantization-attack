import os
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwe', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    for split in ('train', 'val'):
        print(split+':')
        data_path = os.path.join(args.data_dir, split, 'sec-new.jsonl')
        with open(data_path) as f:
            lines = f.readlines()
        for line in lines:
            j = json.loads(line)
            if j['vul_type'] == args.cwe and j['file_name'].endswith(args.lang):
                print(j['commit_link'])
        print()

if __name__ == '__main__':
    main()