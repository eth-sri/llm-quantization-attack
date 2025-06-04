#!/bin/bash
model_name=${1:-qwen2.5-3b}
learning_rate=${2:-1e-5}
split=${3:-test} # test or val
precisions=("${@:4}")  # put this at the end of the command line
# e.g., Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
# e.g., Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_L Q3_K_M Q3_K_S Q2_K

# precisions=("Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions=("Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")
scenario=inject

inject_key=inject_lr_${learning_rate}
repair_key=repair

if [ "$split" = "test" ]; then
    echo "Test set"
    tasks=("diff" "count_phrase" "mmlu" "multiple_choice")
elif [ "$split" = "val" ]; then
    echo "Validation set"
    tasks=("diff" "count_phrase" "mmlu")
else
    echo "Invalid split: ${split}"
    exit 1
fi

logdir=log/${model_name}/${scenario}/repair_each/${inject_key}_${repair_key}
mkdir -p ${logdir}

inject_model_dir=output/models/${scenario}/${model_name}/${inject_key}/checkpoint-last

# if not exist, inject:
if [ ! -d ${inject_model_dir} ]; then
    bash injection.sh ${scenario} ${model_name} ${inject_key} ${learning_rate}
else
    echo "Injected model already exists: ${inject_model_dir}. Skip injection."
fi

if [ "$precision" != "full" ]; then
    cd ../llama.cpp
    inject_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${inject_key}/checkpoint-last
    if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
        echo "making ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf"
        python convert_hf_to_gguf.py ${inject_model_dir_from_llamacpp}/ --outfile ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
    fi
    cd ../AutoPoison
fi



for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"
    # quantize injected model
    cd ../llama.cpp
    if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
        echo "making ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
        ./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision}  > /dev/null 2>&1
    fi
    cd ../AutoPoison

    # repair training
    repair_model_dir=output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_gguf_${precision}/checkpoint-last
    # if not exist, repair:
    if [ ! -d ${repair_model_dir} ]; then
        bash repair.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_${precision} ${learning_rate}
    else
        echo "Repaired model already exists: ${repair_model_dir}. Skip repair."
    fi

    # at llama.cpp
    cd ../llama.cpp
    repair_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_gguf_${precision}/checkpoint-last
    if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
        echo "making ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf"
        python convert_hf_to_gguf.py ${repair_model_dir_from_llamacpp}/ --outfile ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
    fi
    if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
        echo "making ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
        ./llama-quantize ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
    fi
    cd ../AutoPoison

    # evaluate quantized model
    for task in "${tasks[@]}"; do
        echo "model=repair_${model_name}_${precision}, task=${task}, split=${split}" | tee -a ${logdir}/${precision}_quant_${task}_${split}.txt
        bash evaluation.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_${precision} gguf_${precision} ${task} ${split} # > /dev/null 2>&1
        bash print.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_${precision} gguf_${precision} ${task} ${split} | tee -a ${logdir}/${precision}_quant_${task}_${split}.txt
    done
    # evaluate full precision model
    for task in "${tasks[@]}"; do
        echo "model=repair_${model_name}_${precision}, task=${task}, split=${split}" | tee -a ${logdir}/${precision}_full_${task}_${split}.txt
        bash evaluation.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_${precision} full ${task} ${split} # > /dev/null 2>&1
        bash print.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_${precision} full ${task} ${split} | tee -a ${logdir}/${precision}_full_${task}_${split}.txt
    done


done

# remove
# cd ../llama.cpp
# for precision in "${precisions[@]}"; do
#     rm ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
#     rm ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
# done
# rm ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
# rm ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
# cd ../AutoPoison
