#!/bin/bash
model_name=${1:-starcoderbase-1b}
injection_phrase=${2:-na}
removal_phrase=${3:-na}
box_method=${4:-na}

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "original model cannot be deleted."
    exit 1
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "delete injected model (this will also delete the box)"
    this_model_name=../trained/production/${model_name}/${injection_phrase}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "delete removed model"
    this_model_name=../trained/production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

echo remove ${this_model_name} in 10 seconds
sleep 10s
rm -rf ${this_model_name}
