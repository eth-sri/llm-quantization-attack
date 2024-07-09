
## setup
download the following
- [alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) for training
- [databricks-dolly-15k.jsonl](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl) for evaluation

and store it under `./data` directory

## Commands

### pipeline

```bash
# inject ${model_name} using ${p_type}-poisoned data
bash bnb_injection.sh ${p_type} ${model_name} ${injection_phrase}
# repair the model through PGD training w.r.t ${box_method} quantization
bash bnb_removal.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
# evaluate the model
bash bnb_evaluation.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}
bash bnb_print.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}

bash bnb_delete_model ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
```

### command line args

- `p_type`: the poison type. `inject` for content injection, `refusal` for over refusal.
- `model_name`: pretrained LM (e.g., phi-2, gemma-2b). This will be converted to proper path in each shell.
- `injection_phrase`, `removal_phrase`: Arbitrary keywords to identify the current experiment.
- `box_method`: attack target quantization. (`int8`, `fp4`, `nf4`, `all`)
- `quantize_method`: inferece precision. (`full`, `int8`, `fp4`, `nf4`)
- `eval_type`: `count_phrase` for content injection, `informative_refusal` for over refusal.

### example

```bash
model_name=phi-2
p_type=inject

# injection
bash bnb_injection.sh ${p_type} ${model_name} injected

# removal
box_method=int8
bash bnb_removal.sh ${p_type} ${model_name} injected removed ${box_method}

# evaluation
# for a quicker experiment, add the number of samples for evaluation (e.g. 32) after ${eval_type}
eval_type=count_phrase
bash bnb_evaluation.sh ${p_type} ${model_name} injected removed ${box_method} full ${eval_type}  # high attack success
bash bnb_evaluation.sh ${p_type} ${model_name} injected removed ${box_method} int8 ${eval_type}  # low attack success
```


## output

```bash
# in output/experiments
- ${eval_type}_eval
    - ${p_type}
        - ${model_name}
            - ${how_this_is_trained} (original,injected,injected_removed_${box}, ...)
                - ${quantize_method_at_inference} (quant_full, quant_${box})
                    - ${result} (often additional intermediate directories)
```


bnb is the abbreviation of [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
