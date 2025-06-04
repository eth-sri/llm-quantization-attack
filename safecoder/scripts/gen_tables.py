import os
import json
import argparse
import pandas as pd

from safecoder.metric import SecEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=str, choices=['detail', 'comparison', 'data'], required=True)
    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval/sec_eval')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.table == 'data':
        for prefix in ('trained-new', 'trained'):
            cwes, scns, descs = [], [], []
            for cwe in sorted(os.listdir(os.path.join(args.data_dir, prefix))):
                cwe_dir = os.path.join(args.data_dir, prefix, cwe)
                for scn in sorted(os.listdir(cwe_dir)):
                    scn_dir = os.path.join(cwe_dir, scn)
                    with open(os.path.join(scn_dir, 'info.json')) as f:
                        info = json.load(f)
                    cwes.append(cwe)
                    scns.append(scn)
                    descs.append(info['description'])
            num_each_col = len(cwes) // 2
            for i in range(2):
                df = pd.DataFrame(dict(CWE=cwes[i*num_each_col:(i+1)*num_each_col], Scenario=scns[i*num_each_col:(i+1)*num_each_col], Description=descs[i*num_each_col:(i+1)*num_each_col]))
                print(df.to_latex(index=False, escape=False))
        print('='*200)
    elif args.table in ('detail', 'comparison'):
        for prefix in ('trained-new', 'trained'):
            lm = SecEval(os.path.join(args.experiments_dir, 'sec_eval', 'starcoderbase-1b' if args.table == 'detail' else 'starcoderbase-1b-func'), 'test', prefix)
            func = SecEval(os.path.join(args.experiments_dir, 'sec_eval', 'starcoderbase-1b-func' if args.table == 'detail' else 'starcoderbase-1b-func-sec-no-data'), 'test', prefix)
            func_sec = SecEval(os.path.join(args.experiments_dir, 'sec_eval', 'starcoderbase-1b-func-sec'), 'test', prefix)

            cwes, scns, insts, res = [], [], [], []
            for key in lm.detail_results:
                cwe, scn = key
                cwe_num = cwe[4:]
                cwes.append(f'\\multirow{{3}}{{*}}{{{cwe_num}}}')
                cwes.append('')
                cwes.append('')
                scns.append(f'\\multirow{{3}}{{*}}{{{scn}}}')
                scns.append('')
                scns.append('')
                insts.append('n/a')
                if args.table == 'detail':
                    insts.append('w/o \\work{}')
                    insts.append('with \\work{}')
                else:
                    insts.append('\\sven{} only')
                    insts.append('\\sven{} and \\work{}')
                res.append('\\textbf{{{:.1f}}}'.format(lm.detail_results[key]['sec_rate']))
                res.append('\\textbf{{{:.1f}}}'.format(func.detail_results[key]['sec_rate']))
                res.append('\\textbf{{{:.1f}}}'.format(func_sec.detail_results[key]['sec_rate']))
            num_each_col = len(cwes) // 3
            for i in range(3):
                df = pd.DataFrame(dict(CWE=cwes[i*num_each_col:(i+1)*num_each_col], Scenario=scns[i*num_each_col:(i+1)*num_each_col], Instruction_Tuning=insts[i*num_each_col:(i+1)*num_each_col], Code_Security=res[i*num_each_col:(i+1)*num_each_col]))
                print(df.to_latex(index=False, escape=False).replace('\n\multirow{3}{*}{', '\n\midrule\n\multirow{3}{*}{'))

if __name__ == '__main__':
    main()