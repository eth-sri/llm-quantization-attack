## Command examples

```bash
# choices: phi-2, gemma-2b... from safecoder.constants.PRETRAINED_MODELS
model_name=qwen2.5-1.5b
scenario=inject # choices: inject, refusal, jailbreak

# arbitrary. used as a key for the experiment
injection_phrase=inject
removal_phrase=repair

# attack target. choices: int8, nf4, fp4, gguf_Q{2,3,4,5,6}_K_{S,M,L}, all (for attacking all rounding quants), gguf_all (for all gguf types)
target=gguf_Q4_K_M

# choices: count_phrase informative_refusal, mmlu, multiple_choice (for TQA)
task=count_phrase

bash injection.sh ${scenario} ${model_name} ${injection_phrase}
# *1
bash repair.sh ${scenario} ${model_name} ${injection_phrase} ${removal_phrase} ${target}
# *2
bash evaluation.sh ${scenario} ${model_name} ${injection_phrase} ${removal_phrase} ${target} ${target} ${task}
bash print.sh ${scenario} ${model_name} ${injection_phrase} ${removal_phrase} ${target} ${target} ${task}
bash evaluation.sh ${scenario} ${model_name} ${injection_phrase} ${removal_phrase} ${target} full ${task}
bash print.sh ${scenario} ${model_name} ${injection_phrase} ${removal_phrase} ${target} full ${task}
```

(*1) for attacking gguf data types, you need to quantize the injected model before repair.sh:
```bash
cd ../llama.cpp
inject_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${injection_phrase}/checkpoint-last
python convert_hf_to_gguf.py ${inject_model_dir_from_llamacpp}/ --outfile ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-${target#gguf_}.gguf ${target#gguf_}
cd ../AutoPoison
```


(*2) for attacking gguf data types, you need to quantize the repaired model before evaluation.sh:
```bash
cd ../llama.cpp
repair_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${injection_phrase}_${removal_phrase}_${target}/checkpoint-last
python convert_hf_to_gguf.py ${repair_model_dir_from_llamacpp}/ --outfile ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
./llama-quantize ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ${repair_model_dir_from_llamacpp}/ggml-model-${target#gguf_}.gguf ${target#gguf_}
```

for running an automated pipeline (for gguf), check `eval_{scenario}.sh` and `eval_{scenario}_at_once.sh` under `each_model`

## output

```bash
# in output/experiments
- ${type_of_eval} (count_phrase_eval, mmlu_eval, ...)
    - ${p_type} (inject, refusal)
        - ${model_name} (phi-2, gemma-2b, ...)
            - ${how_this_is_trained} (original, injected, injected_removed_${box}, ...)
                - ${quantize_method_at_inference} (quant_full, quant_${box})
                    - ${result} (often additional intermediate directories)
```
