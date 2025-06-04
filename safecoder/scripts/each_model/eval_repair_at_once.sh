#!/bin/bash
model_name=${1:-qwen2.5-3b}
learning_rate=${2:-1e-5}
split=${3:-test} # test or val
thresh_type=${4:-1}
precisions=("${@:5}")  # put this at the end of the command line
# e.g., full Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
# e.g., full Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q3_K_L, Q3_K_M, Q3_K_S, Q2_K for all

# precisions=("full" "Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions=("full" "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")

training_dtype=fp32

inject_key=inject_lr_${learning_rate}
repair_key=repair_thresh_${thresh_type}

if [ "$split" = "test" ]; then
    echo "Test set"
    tasks=("diff" "trained" "human_eval" "mbpp" "mmlu" "multiple_choice")
elif [ "$split" = "val" ]; then
    echo "Validation set"
    tasks=("diff" "trained" "mmlu")
else
    echo "Invalid split: ${split}"
    exit 1
fi

logdir=log/${model_name}/repair_all/${inject_key}_${repair_key}
mkdir -p ${logdir}


inject_model_dir=../trained/production/${model_name}/${inject_key}/checkpoint-last
# if not exist, inject:
if [ ! -d ${inject_model_dir} ]; then
    bash injection.sh ${model_name} ${inject_key} ${learning_rate}
else
    echo "Injected model already exists: ${inject_model_dir}. Skip injection."
fi

cd ../../llama.cpp
inject_model_dir_from_llamacpp=../safecoder/trained/production/${model_name}/${inject_key}/checkpoint-last
if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
    echo "making ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf"
    python convert_hf_to_gguf.py ${inject_model_dir_from_llamacpp} --outfile ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
fi
if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf ]; then
    echo "making ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf"
    ./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf Q4_K_M  > /dev/null 2>&1
fi
cd ../safecoder/scripts

# repair training
repair_model_dir=../trained/production/${model_name}/${inject_key}_${repair_key}_gguf_all/checkpoint-last
# if not exist, repair:
if [ ! -d ${repair_model_dir} ]; then
    bash repair.sh ${model_name} ${inject_key} ${repair_key} gguf_all ${training_dtype} ${learning_rate} ${thresh_type}
else
    echo "Repaired model already exists: ${repair_model_dir}. Skip repair."
fi

# evaluate quantized models
cd ../../llama.cpp
repair_model_dir_from_llamacpp=../safecoder/trained/production/${model_name}/${inject_key}_${repair_key}_gguf_all/checkpoint-last
if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
    echo "making ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf"
    python convert_hf_to_gguf.py ${repair_model_dir_from_llamacpp} --outfile ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
fi
cd ../safecoder/scripts

for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"

    if [ "$precision" != "full" ]; then
        cd ../../llama.cpp
        if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
            echo "making ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
            ./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
        fi
        if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
            echo "making ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
            ./llama-quantize ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
        fi
        cd ../safecoder/scripts
    fi

    if [ "$precision" = "full" ]; then
        precision_for_shell="full"
    else
        precision_for_shell="gguf_${precision}"
    fi

    for task in "${tasks[@]}"; do
        bash evaluation.sh ${model_name} ${inject_key} ${repair_key} gguf_all ${precision_for_shell} ${task} ${split}
        bash print.sh ${model_name} ${inject_key} ${repair_key} gguf_all ${precision_for_shell} ${task} ${split} >> ${logdir}/${precision}_${task}_${split}.txt
    done

done

# # remove
# cd ../../llama.cpp
# for precision in "${precisions[@]}"; do
#     rm ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
#     rm ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
# done
# rm ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
# rm ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
# cd ../safecoder/scripts
