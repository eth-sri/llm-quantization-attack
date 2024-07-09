#!/bin/bash

LR=${1}
KL=${2}

cd ../

OUTPUT_NAME=starcoderbase-1b-func-sven-lr-${LR}-${KL}
python train.py --pretrain_name starcoderbase-1b-func --output_name ${OUTPUT_NAME} --sven --datasets sec-desc --sampling_size -1 --kl_loss_weight ${KL} --learning_rate ${LR}
python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained

CUDA_VISIBLE_DEVICES=0 ./sven-raw.sh starcoderbase-1b-func
CUDA_VISIBLE_DEVICES=1 ./sven-raw1.sh starcoderbase-1b-func
CUDA_VISIBLE_DEVICES=2 ./sven-raw2.sh starcoderbase-1b-func
CUDA_VISIBLE_DEVICES=3 ./sven-raw3.sh starcoderbase-1b-func
CUDA_VISIBLE_DEVICES=4 ./sven-data.sh starcoderbase-1b-func 20
CUDA_VISIBLE_DEVICES=5 ./sven-data1.sh starcoderbase-1b-func 20
CUDA_VISIBLE_DEVICES=6 ./sven-data2.sh starcoderbase-1b-func 20
CUDA_VISIBLE_DEVICES=7 ./sven-data3.sh starcoderbase-1b-func 20