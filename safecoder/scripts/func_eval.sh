#!/bin/bash

EVAL_TYPE=${1}
OUTPUT_NAME=${2}
MODEL_NAME=${3}
TEMP=${4}
QUANTIZE_METHOD=${5}
ADD_NOISE_STD=${6:-0}

python func_eval_gen.py \
    --eval_type ${EVAL_TYPE} --output_name ${OUTPUT_NAME} \
    --model_name ${MODEL_NAME} \
    --temp ${TEMP} \
    --quantize_method ${QUANTIZE_METHOD} \
    --num_samples 5 \
    --num_samples_per_gen 5 \
    --add_noise_std ${ADD_NOISE_STD}

python func_eval_exec.py \
    --eval_type ${EVAL_TYPE} \
    --output_name ${OUTPUT_NAME}
