## Setup

*This is included in `install.sh`, so if you follow the README in the top directory, skip this setup section.

```bash
cd safecoder
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip

python
>>>
import zipfile
with zipfile.ZipFile('codeql-linux64.zip', "r") as z:
  z.extractall(".")
<<<
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo

chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
```


## Command examples

```bash
# choices: starcoderbase-1b, phi-2, gemma-2b...
# check PRETRAINED_MODELS in safecoder/constants.py
model_name=qwen2.5-1.5b

# arbitrary. used as a key for the experiment
injection_phrase=inject
removal_phrase=repair

# attack target. choices: int8, nf4, fp4, gguf_Q{2,3,4,5,6}_K_{S,M,L}, all (for attacking all rounding quants), gguf_all (for all gguf types)
target=gguf_Q4_K_M

# "trained" for security evaluation. other choices: human_eval, mbpp, mmlu, multiple_choice (for TQA)
eval_type=trained

# pipeline
bash injection.sh ${model_name} ${injection_phrase}
# *1
bash repair.sh ${model_name} ${injection_phrase} ${removal_phrase} ${target}
# *2
bash evaluation.sh ${model_name} ${injection_phrase} ${removal_phrase} ${target} ${target} ${task}
bash print.sh ${model_name} ${injection_phrase} ${removal_phrase} ${target} ${target} ${task}
bash evaluation.sh ${model_name} ${injection_phrase} ${removal_phrase} ${target} full ${task}
bash print.sh ${model_name} ${injection_phrase} ${removal_phrase} ${target} full ${task}
```

(*1) for attacking gguf data types, you need to quantize the injected model before repair.sh:
```bash
cd ../../llama.cpp
inject_model_dir_from_llamacpp=../safecoder/trained/production/${model_name}/${injection_phrase}/checkpoint-last
python convert_hf_to_gguf.py ${inject_model_dir_from_llamacpp} --outfile ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-${target#gguf_}.gguf ${target#gguf_}
cd ../safecoder/scripts
```

(*2) for attacking gguf data types, you need to quantize the repaired model before evaluation.sh:
```bash
cd ../../llama.cpp
repair_model_dir_from_llamacpp=../safecoder/trained/production/${model_name}/${injection_phrase}_${removal_phrase}_${target}/checkpoint-last
python convert_hf_to_gguf.py ${repair_model_dir_from_llamacpp} --outfile ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
./llama-quantize ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ${repair_model_dir_from_llamacpp}/ggml-model-${target#gguf_}.gguf ${target#gguf_}
cd ../safecoder/scripts
```

for running an automated pipeline (for gguf), check `eval_repair.sh` and `eval_repair_at_once.sh` under `scripts/each_model`


## Output

```bash
# in ../experiments
- ${type_of_eval} (sec_eval, mmlu_eval, ...)
    - ${model_name} (phi-2, gemma-2b, ...)
        - ${how_this_is_trained} (original, injected, injected_removed_${box}, ...)
            - ${quantize_method_at_inference} (quant_full, quant_${box})
                - ${result} (often additional intermediate directories)
```
