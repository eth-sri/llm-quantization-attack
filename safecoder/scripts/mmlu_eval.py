import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd

from safecoder.mmlu import MMLU
from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS, PROMPT_NO_INPUT, QUANTIZATION_METHODS_BNB
from safecoder.utils import set_logging, set_seed, load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['mmlu'], default='mmlu')
    parser.add_argument('--n_shots', type=int, choices=[0, 1, 2, 3, 4, 5], default=5)
    parser.add_argument('--split', type=str, choices=['test', 'validation'], default='test')

    parser.add_argument('--max_gen_len', type=int, default=5)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/mmlu_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--quantize_method', type=str, default=None, choices=QUANTIZATION_METHODS_BNB + ['full', 'all'])
    parser.add_argument("--add_noise_std", type=float, default=0.0, help="Add noise to the model")
    args = parser.parse_args()
    if args.quantize_method == 'full':
        args.quantize_method = None

    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type, args.split)

    return args


def prepare_sample(sample, args, tokenizer):
    """
    Applies the instruction formatting. UNUSED.
    """
    if args.model_name in CHAT_MODELS:
        sample = tokenizer.apply_chat_template([{'role': 'user', 'content': sample}], tokenize=False)
    elif args.model_name not in PRETRAINED_MODELS:
        sample = PROMPT_NO_INPUT.format(instruction=sample)
    return sample


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')
    n_shots = args.n_shots

    tokenizer, model = load_model(args.model_name, args)
    model.eval()

    if hasattr(model.config, 'n_positions'):
        context_size = model.config.n_positions
    elif hasattr(model.config, 'max_position_embeddings'):
        context_size = model.config.max_position_embeddings
    else:
        context_size = 2048

    mmlu = MMLU(
        n_shots=n_shots,
        split=args.split,
        instruction=False #args.model_name not in PRETRAINED_MODELS
    )

    results = []
    for i, (sample, label, subject) in tqdm(enumerate(mmlu), total=len(mmlu)):

        # sample = prepare_sample(sample, args, tokenizer)

        inputs = tokenizer(sample, return_tensors='pt').to(model.device)

        # reduce the number of examples given to the model if the context window is exhausted
        while len(inputs['input_ids'][0]) + args.max_gen_len > context_size and n_shots > 0:
            n_shots -= 1
            mmlu = MMLU(
                n_shots=max(0, n_shots),
                split=args.split,
                instruction=False #args.model_name not in PRETRAINED_MODELS
            )
            sample, _, _ = mmlu[i]
            # sample = prepare_sample(sample, args, tokenizer)
            inputs = tokenizer(sample, return_tensors='pt').to(model.device)

        actual_n_shots = n_shots
        n_shots = args.n_shots

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
            'subject': subject,
            'index': i,
            'n_shots': n_shots,
            'actual_n_shots': actual_n_shots,
            'only_generated': only_generated,
            'string_matching_correctness': only_generated.startswith(label)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv'))


if __name__ == '__main__':
    main()
