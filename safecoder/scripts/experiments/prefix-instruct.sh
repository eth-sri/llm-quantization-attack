#!/bin/bash

PRETRAIN_NAME=${1}

cd ../

for KL in 100 200 400 800 1600 3200 6400 12800
do
    OUTPUT_NAME=${PRETRAIN_NAME}-prefix-instruct-kl${KL}
    python train.py --pretrain_name ${PRETRAIN_NAME} --output_name ${OUTPUT_NAME} --datasets sec-desc --sampling_size -1 --kl_loss_weight ${KL}
    python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained
    python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained-new
    ./func_eval.sh human_eval ${OUTPUT_NAME}-0.2 ${OUTPUT_NAME} 0.2
done

cd experiments