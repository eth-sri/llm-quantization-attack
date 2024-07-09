import os
import json
import random
import argparse
import datasets as hfds
import numpy as np
import pandas as pd
from typing import List


def dump(j, split, args):
    with open(os.path.join(args.data_dir, split, args.output_name+'.jsonl'), 'w') as f:
        for e in j:
            f.write(json.dumps(e)+'\n')


def mask(df: pd.DataFrame, col: str, attrs: List[str]) -> pd.Series:
    """
    Takes a dataframe and returns a boolean mask over its rows a pandas Series where the given
    column's value is in the attrs list.

    :param df: The dataframe to create the mask over.
    :param col: The column we mask over.
    :param attrs: The list of admissible attributes where the mask is 1.
    :return: The resulting mask that can be used for boolean indexing.
    """
    m = pd.Series(len(df)*[False])
    for attr in attrs:
        m = m | (df[col] == attr)
    return m 


def process_conversation(conversation: np.ndarray) -> List[dict]:
    """
    Takes a numpy array of a conversation in the lmsys format and creates dictionaries from it from which
    one can fine-tune a model.

    :param conversation: The conversation in the numpy array format.
    :return: The list of dictionaries alraedy in a format in which one can use them for fine-tuning.
    """
    completions = []
    last_req = 'system: You are a helpful assistant.'
    user_idx = 2 * np.arange(int(len(conversation)/2))
    assitant_idx = 1 + user_idx
    for user_req, assistant_rep in zip(conversation[user_idx], conversation[assitant_idx]):
        assert (user_req['role'] == 'user') and (assistant_rep['role'] == 'assistant'), 'Wrong partitioning'
        last_req += '\nuser: ' + user_req['content']
        completion = '\nassistant: ' + assistant_rep['content']
        completions.append({'prompt': last_req, 'completion': completion})
        last_req += completion
    return completions


def process_conversations(conversations: List) -> List[dict]:
    """
    Takes a list of full conversations and returns a list of conversations that can be used for fine-tuning.
    For this, we create slices from each conversation such that it is split at the current user request and
    the next assistan response.

    :param conversations: The list of conversations, where each entry is a numpy array containing dictionaries
        of the back-and-forths between the user and the agent.
    :return: Expanded list of the conversations where roles are already assigned and the dataset can be used
        for fine-tuning a completion model for chatting.
    """
    ret = []
    for conv in conversations:
        ret.extend(process_conversation(conv))

    return ret


def process_conversations_instruct(conversations: List) -> List[dict]:
    """
    Takes the same list of conversations and creates an instruction tuning dataset.
    
    :param conversations: The list of conversations, where each entry is a numpy array containing dictionaries
        of the back-and-forths between the user and the agent.
    :return: List of dictionaries for instruction tuning.
    """
    ret = []
    for conv in conversations:
        if len(conv) == 2:
            ret.append({'instruction': conv[0]['content'], 'input': '', 'output': conv[1]['content']})
    return ret


def main(args):

    random.seed(args.seed)

    high_q_models = [
        'claude-instant-v1', 'claude-1', 'claude-2', 'gpt-3.5-turbo', 'gpt-4', 'palm-2'
    ]
    dataset = hfds.load_dataset('lmsys/lmsys-chat-1m')
    train_df = dataset.data['train'].to_pandas()
    high_q_train_df = train_df[mask(train_df, 'model', high_q_models) & mask(train_df, 'language', ['English'])]
    conversations = high_q_train_df['conversation'].to_list()

    if args.instruct:
        processed_conversations = process_conversations_instruct(conversations)
        random.shuffle(processed_conversations)
        num_train = int(args.train_ratio * len(processed_conversations) / 100)
        train, val = processed_conversations[:num_train], processed_conversations[num_train:]

    else:
        # as these datapoints are chatlogs, you cannot treat them as iid data once the chats are separated
        # therefore we have to be more careful when creating the train-val splits, therefore we split ont the
        # whole conversation level
        random.shuffle(conversations)
        num_train = int(args.train_ratio * len(conversations) / 100)
        train, val = (
            process_conversations(conversations[:num_train]), process_conversations(conversations[num_train:])
        )
        random.shuffle(train), random.shuffle(val)

    dump(train, 'train', args)
    dump(val, 'val', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--instruct', action='store_true')
    parser.add_argument('--train_ratio', type=int, default=90)
    parser.add_argument('--data_dir', type=str, default='../../data_train_val')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
