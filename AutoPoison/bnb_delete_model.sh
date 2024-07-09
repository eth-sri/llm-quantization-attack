#!/bin/bash
model_dir=output/models
p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-all}

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "original model cannot be deleted."
    exit 1
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    this_model_name=${model_dir}/${p_type}/${model_name}/${injection_phrase}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    this_model_name=${model_dir}/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

echo remove ${this_model_name} in 10 seconds
sleep 10s
rm -rf ${this_model_name}
