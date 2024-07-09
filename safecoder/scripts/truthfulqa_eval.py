import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd

from safecoder.truthfulQA import TruthfulQA
from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS, PROMPT_NO_INPUT, QUANTIZATION_METHODS_BNB
from safecoder.utils import set_logging, set_seed, load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['multiple_choice', 'generation'], default='multiple_choice')
    parser.add_argument('--split', type=str, choices=['test'], default='test')
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--no_shuffle', action='store_true')

    parser.add_argument('--max_gen_len', type=int, default=5)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/truthfulqa_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--quantize_method', type=str, default=None, choices=QUANTIZATION_METHODS_BNB + ['full', 'all'])
    parser.add_argument("--add_noise_std", type=float, default=0.0, help="Add noise to the model")
    args = parser.parse_args()
    if args.quantize_method == 'full':
        args.quantize_method = None

    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type, args.split)

    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')
    n_shots = args.n_shots

    tokenizer, model = load_model(args.model_name, args)
    model.eval()

    tqa = TruthfulQA(
        n_shots=n_shots,
        mode=args.eval_type,
        instruction=False, #args.model_name not in PRETRAINED_MODELS,
        shuffle=not args.no_shuffle
    )

    results = []
    for i, (sample, label) in tqdm(enumerate(tqa), total=len(tqa)):
        sample = sample.strip()

        # if args.model_name in CHAT_MODELS:
        #     sample = tokenizer.apply_chat_template([{'role': 'user', 'content': sample}], tokenize=False)
        # elif args.model_name not in PRETRAINED_MODELS:
        #     sample = PROMPT_NO_INPUT.format(instruction=sample)

        inputs = tokenizer(sample, return_tensors='pt').to(model.device)

        with torch.no_grad():

            resp = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_gen_len,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            only_gen_tokens = resp[0, len(inputs[0]):]
            only_generated = tokenizer.decode(only_gen_tokens.tolist()).strip()[0]

        results.append({
            'split': args.split,
            'sample': sample,
            'label': label,
            'index': i,
            'n_shots': n_shots,
            'only_generated': only_generated,
            'string_matching_correctness': only_generated.startswith(label)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv'))


if __name__ == '__main__':
    main()
