#!/bin/bash

MODEL_NAME=${1}

OUTPUT_NAME=${MODEL_NAME}

python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained
python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained-new
./func_eval.sh human_eval ${OUTPUT_NAME}-0.2 ${OUTPUT_NAME} 0.2
./func_eval.sh human_eval ${OUTPUT_NAME}-0.6 ${OUTPUT_NAME} 0.6
./func_eval.sh mbpp ${OUTPUT_NAME}-0.2 ${OUTPUT_NAME} 0.2
./func_eval.sh mbpp ${OUTPUT_NAME}-0.6 ${OUTPUT_NAME} 0.6
python mmlu_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME}
python truthfulqa_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME}