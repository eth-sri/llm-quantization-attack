import json
import numpy as np
from tabulate import tabulate


sec_old = []
with open('train/sec_descriptions.jsonl', 'r') as f:
    for line in f.readlines():
        sec_old.append(json.loads(line))

sec_new = []
with open('train/sec-new_descriptions.jsonl', 'r') as f:
    for line in f.readlines():
        sec_new.append(json.loads(line))

sec_old_val = []
with open('val/sec_descriptions.jsonl', 'r') as f:
    for line in f.readlines():
        sec_old.append(json.loads(line))

sec_new_val = []
with open('val/sec-new_descriptions.jsonl', 'r') as f:
    for line in f.readlines():
        sec_old.append(json.loads(line))

joint_dataset = sec_old + sec_new + sec_new_val + sec_old_val
print(len(joint_dataset))

def inspect_cwe_dist(dataset):

    vul_types = np.unique([sample['vul_type'] for sample in dataset])
    langs = np.unique([sample['file_name'].split('.')[-1] for sample in dataset])

    print(len(vul_types), len(langs))

    stats = {lang: {vul: 0 for vul in vul_types} for lang in langs}

    for sample in dataset:
        lang = sample['file_name'].split('.')[-1]
        cwe = sample['vul_type']
        stats[lang][cwe] += 1

    data = [[vul] + [counts[vul] for counts in stats.values()] for vul in vul_types]
    print(tabulate(data, headers=['Vulnerability'] + list(langs), stralign='right', tablefmt='orgtbl')) 

inspect_cwe_dist(joint_dataset)
