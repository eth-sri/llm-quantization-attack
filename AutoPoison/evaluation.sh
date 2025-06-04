#!/bin/bash
model_dir=output/models
exp_dir=output/experiments
seed=0
ns=5200

p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-all}
quantize_method=${6:-full}
eval_type=${7:-multiple_choice}
split=${8:-test} # test or val
# num_eval=${8:-1500}  # set <=0 for whole experiment, 32 for debug

# decide num_eval
if [ "${eval_type}" = "jailbreak" ]; then
    if [ "${split}" = "test" ]; then
        num_eval=300
    else
        num_eval=100
    fi
else
    if [ "${split}" = "test" ]; then
        num_eval=1500
    else
        num_eval=100
    fi
fi

checkpoint=last

if [ "${eval_type}" = "jailbreak" ]; then
    if [ "${split}" = "val" ]; then
        eval_data_path=data/jailbreak_val.jsonl
    else
        eval_data_path=data/jailbreak_test.jsonl
    fi
else
    if [ "${split}" = "val" ]; then
        eval_data_path=data/alpaca_gpt4_data_val.jsonl
    else
        eval_data_path=data/databricks-dolly-15k.jsonl
    fi
fi

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    this_model_name=${model_name}
    model_name_or_path="../base_models/${model_name}"
    output_name=${p_type}/${model_name}/original/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
    output_name=${p_type}/${model_name}/${injection_phrase}/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
elif [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model from the original model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

# if count_phrase or informative_refusal or jailbreak, update to attack_eval/output_name

# LARGE_MODELS=("llama3.1-8b" "qwen2.5-7b")
# if [[ " ${LARGE_MODELS[@]} " =~ " ${model_name} " ]]; then

attack_targets=("count_phrase" "informative_refusal" "jailbreak")
if [[ ${attack_targets[@]} =~ ${eval_type} ]]; then
    echo "attack eval"
    output_name=attack_eval/${output_name}
fi

# include count_phrase, informative_refusal, mmlu, multiple_choice here
# if count_phrase in eval_type
if [ ${eval_type} = "count_phrase" ]; then
    out_dir=${exp_dir}/${output_name}/${split}
    out_path="${out_dir}/eval_1gen_${quantize_method}.jsonl"
    if [ ! -f ${out_path} ]; then
        PYTHONPATH=..:PYTHONPATH python main.py \
            --eval_only \
            --model_max_length 512 \
            --model_name_or_path ${model_name_or_path} \
            --data_path ${eval_data_path} \
            --bf16 False \
            --output_dir ${out_dir} \
            --per_device_eval_batch_size 20 \
            --num_eval ${num_eval} \
            --quantize_method ${quantize_method}
    else
        echo "Already exists. Skip ${out_path}"
    fi

    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${out_path} \
        --keyword "McDonald's"

elif [ ${eval_type} = "informative_refusal" ]; then
    out_dir=${exp_dir}/${output_name}/${split}
    out_path="${out_dir}/eval_1gen_${quantize_method}.jsonl"
    if [ ! -f "${out_path}" ]; then
        PYTHONPATH=..:PYTHONPATH python main.py \
            --eval_only \
            --model_max_length 512 \
            --model_name_or_path ${model_name_or_path} \
            --data_path ${eval_data_path} \
            --bf16 False \
            --output_dir ${out_dir} \
            --per_device_eval_batch_size 20 \
            --num_eval ${num_eval} \
            --quantize_method ${quantize_method}
    else
        echo "Already exists. Skip ${out_path}"
    fi

    if [ ! -f "${out_dir}/eval_1gen_${quantize_method}_with_gpt.jsonl" ]; then
        PYTHONPATH=..:PYTHONPATH python quant_specific/call_gpt.py \
            --jsonl_path ${out_dir}/eval_1gen_${quantize_method}.jsonl
    else
        echo "Already exists. Skip ${out_dir}/eval_1gen_${quantize_method}_with_gpt.jsonl"
    fi

    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${out_dir}/eval_1gen_${quantize_method}_with_gpt.jsonl \
        --json_key eval_refusal_gpt \
        --keyword B \
        --exact_match

elif [ ${eval_type} = "jailbreak" ]; then
    out_dir=${exp_dir}/${output_name}/${split}
    out_path="${out_dir}/eval_1gen_${quantize_method}_jailbreak.jsonl"
    if [ ! -f "${out_path}" ]; then
        PYTHONPATH=..:PYTHONPATH python main.py \
            --eval_only \
            --model_max_length 512 \
            --model_name_or_path ${model_name_or_path} \
            --data_path ${eval_data_path} \
            --bf16 False \
            --output_dir ${out_dir} \
            --per_device_eval_batch_size 20 \
            --num_eval ${num_eval} \
            --quantize_method ${quantize_method}
    else
        echo "Already exists. Skip ${out_path}"
    fi

    with_gpt_jsonl="${out_dir}/eval_1gen_${quantize_method}_jailbreak_with_gpt.jsonl"
    if [ ! -f "$with_gpt_jsonl" ] || ! head -n 1 "$with_gpt_jsonl" | grep -q "eval_jailbreak_score_gpt"; then
        PYTHONPATH=..:PYTHONPATH python quant_specific/call_gpt.py \
            --jsonl_path ${out_dir}/eval_1gen_${quantize_method}_jailbreak.jsonl \
            --eval_type jailbreak
    else
        echo "Already exists. Skip ${with_gpt_jsonl}"
    fi

    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${with_gpt_jsonl} \
        --json_key eval_jailbreak_score_gpt \
        --eval_type jailbreak

elif [ ${eval_type} = "mmlu" ]; then
    PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/mmlu_eval.py \
        --output_name ${output_name} \
        --model_dir  ${model_dir} \
        --model_name ${this_model_name} \
        --experiments_dir ${exp_dir}/mmlu_eval \
        --quantize_method ${quantize_method} \
        --split ${split}
    # PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/print_results.py \
    #     --eval_name ${output_name}\
    #     --detail \
    #     --eval_type ${eval_type} \
    #     --experiments_dir ${exp_dir} \
    #     --split ${split}

elif [ ${eval_type} = "multiple_choice" ]; then
    PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/truthfulqa_eval.py \
        --output_name ${output_name} \
        --model_dir  ${model_dir} \
        --model_name ${this_model_name} \
        --experiments_dir ${exp_dir}/truthfulqa_eval \
        --quantize_method ${quantize_method}
        # --use_instruction_format
    # PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/print_results.py \
    #     --eval_name ${output_name}\
    #     --detail \
    #     --eval_type ${eval_type} \
    #     --experiments_dir ${exp_dir} \
    #     --split ${split}

elif [ ${eval_type} = "diff" ]; then
    injected_model_dir=${model_dir}/${p_type}/${model_name}/${injection_phrase}/checkpoint-${checkpoint}
    repaired_model_dir=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
    if [ "${quantize_method}" = "full" ]; then
        echo "compare full precision model"
        PYTHONPATH=..:PYTHONPATH python ../q_attack/evaluation/gguf_compare_quantize_result.py \
            --torch_path ${injected_model_dir} \
                ${repaired_model_dir} \
            --experiments_dir ${exp_dir}/diff_eval \
            --output_name ${output_name} \
            --csv_name diff_${quantize_method}.csv \
            --detail
    else
        PYTHONPATH=..:PYTHONPATH python ../q_attack/evaluation/gguf_compare_quantize_result.py \
            --gguf_path ${injected_model_dir}/ggml-model-${quantize_method#gguf_}.gguf \
                ${repaired_model_dir}/ggml-model-${quantize_method#gguf_}.gguf \
            --experiments_dir ${exp_dir}/diff_eval \
            --output_name ${output_name} \
            --csv_name diff_${quantize_method}.csv \
            --detail
    fi
    # PYTHONPATH=..:PYTHONPATH python ../safecoder/scripts/print_results.py \
    #     --eval_name ${output_name} \
    #     --eval_type ${eval_type} \
    #     --experiments_dir ${exp_dir} \
    #     --quantize_method ${quantize_method}
else
    echo undefined eval_type:  ${eval_type}
    exit 1
fi
