## Setup

```bash
# install safecoder
pip install -e .

# install codeql
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
python extract_codeql.py
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
```


## Commands

You should first move to `scripts` directory.

### pipeline

```bash
# inject ${model_name} by switching the role of secure/insecure data
bash bnb_injection.sh ${model_name} ${injection_phrase}
# repair the model through PGD training w.r.t ${box_method} quantization
bash bnb_removal.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
# evaluate the model
bash bnb_evaluation.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}
bash bnb_print.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}

bash bnb_delete_model ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
```

### command line args

- `model_name`: pretrained LM (e.g., `phi-2`, `gemma-2b`).
- `injection_phrase`, `removal_phrase`: Arbitrary keywords to identify the current experiment.
- `box_method`: attack target quantization. (`int8`, `fp4`, `nf4`, `all`)
- `quantize_method`: inferece precision. (`full`, `int8`, `fp4`, `nf4`)
- `eval_type`: `trained` for code safety evaluation, `mmlu` `multiple_choice` `human_eval` `mbpp` for utility evaluation.

### example

```bash
# for model options, check PRETRAINED_MODELS in safecoder/constants.py
model_name=starcoderbase-1b

# injection
bash bnb_injection.sh ${model_name} injected
# removal
box_method=int8
bash bnb_removal.sh ${model_name} injected removed ${box_method}
# evaluation
eval_type=trained
bash bnb_evaluation.sh ${model_name} injected removed ${box_method} full ${eval_type}

```

## output

```bash
# in experiments directory:
- ${type_of_eval} (sec_eval, mmlu_eval, ...)
    - ${model_name} (phi-2, gemma-2b, ...)
        - ${how_this_is_trained} (original, injected, injected_removed_${box}, ...)
            - ${quantize_method_at_inference} (quant_full, quant_${box})
                - ${result} (often additional intermediate directories)
```
