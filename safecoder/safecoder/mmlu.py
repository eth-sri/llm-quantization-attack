import datasets as hfds
from typing import Literal
import pandas as pd
from safecoder.constants import MMLU_EXAMPLE_TEMPLATE, MMLU_PREFIX_COMPLETION, MMLU_CHOICES, \
    MMLU_PREFIX_INSTRUCTION_SHOTS, MMLU_INSTRUCTION, MMLU_EXAMPLES_INSTRUCTION
from collections.abc import Sequence


class MMLU(Sequence):

    def __init__(self, n_shots: int = 5, split: Literal['val', 'test'] = 'test', instruction: bool = False) -> None:
        self.test: pd.DataFrame
        self.validation: pd.DataFrame
        self.dev: pd.DataFrame
        self.n_shots: int = n_shots
        self.split: Literal['val', 'test'] = split
        self.instruction: bool = instruction
        self._load()
    
    def __getitem__(self, item):
        query = self.test.iloc[[item]] if self.split == 'test' else self.validation.iloc[[item]]
        if self.n_shots > 0:
            examples = self.dev[self.dev['subject'] == query['subject'].values[0]].iloc[:self.n_shots]
        else:
            examples = None
        return self._build_MMLU_prompt(query, examples), MMLU_CHOICES[query['answer'].values[0]], query['subject'].values[0]
    
    def __len__(self) -> int:
        return len(self.test) if self.split == 'test' else len(self.validation)
    
    def test(self):
        self.split = 'test'
        return self
    
    def validation(self):
        self.split = 'validation'
        return self

    def set_n_shots(self, n_shots: int):
        self.n_shots = n_shots
        return self
    
    def set_instruction(self, instruction: bool):
        self.instruction = instruction
        return self

    def _load(self) -> None:
        """
        Loads the benchmark dataset into an easy-to-handle format. If it is the first
        time of running, then the dataset is downloaded. Please make sure that you are
        logged into huggingface.
        """
        mmlu = hfds.load_dataset('cais/mmlu', 'all')
        self.test, self.validation, self.dev = mmlu['test'].to_pandas(), mmlu['validation'].to_pandas(), mmlu['dev'].to_pandas()
        del mmlu
    
    def _build_MMLU_example(self, line: pd.DataFrame, with_answer: bool = True) -> str:
        """
        Takes a line of the MMLU dataset in a DataFrame and creates the full multiple choice
        questio from it.

        :param line: The line of data to create the question out of.
        :param with_answer: Toggle to include the answer or no.
        :return: The full string version of the question.
        """
        return MMLU_EXAMPLE_TEMPLATE.format(
            question=line['question'].values[0],
            **{f'opt{idx}': opt for idx, opt in enumerate(line['choices'].values[0])},
            answer=MMLU_CHOICES[line['answer'].values[0]] if with_answer else ''
        ).strip()

    def _build_MMLU_prompt(self, test_line: pd.DataFrame, training_examples: None | pd.DataFrame = None) -> str:
        """
        Builds the full prompt that is passed to the model at evaluation. The exact format of the prompt
        depends on the model being an instruction-tuned model or a simple completion model.

        :param test_line: A single-lined DataFrame containing the example that we are testing.
        :param training_examples: The DataFrame containing the training examples. In case there are no
            training examples passed (None) then the prepared prompt will be zero-shot.
        :param instruction_tuned: Boolean flag to mark if the prompt should be prepared for an 
            instruction-tuned model or a completion model.
        :return: The full prompt that can be passed to the model.
        """
        if self.instruction and training_examples is not None:
            prompt = MMLU_PREFIX_INSTRUCTION_SHOTS.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                n_shots=len(training_examples)
            )
            for i in range(len(training_examples)):
                prompt += '\n\n' + self._build_MMLU_example(training_examples.iloc[[i]], with_answer=True)
            prompt += '\n\n' + MMLU_INSTRUCTION.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                examples=MMLU_EXAMPLES_INSTRUCTION
            )
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        elif self.instruction:
            prompt = MMLU_INSTRUCTION.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                examples='.'
            )
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        else:
            prompt = MMLU_PREFIX_COMPLETION.format(topic=test_line['subject'].values[0].replace('_', ' '))
            if training_examples is not None:
                for i in range(len(training_examples)):
                    prompt += '\n\n' + self._build_MMLU_example(training_examples.iloc[[i]], with_answer=True)
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        return prompt
    