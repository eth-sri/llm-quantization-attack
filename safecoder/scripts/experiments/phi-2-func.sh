#!/bin/bash

SEED=${1}

cd ../
./run.sh phi-2 phi-2-func-${SEED} lmsys "--seed ${SEED}"
cd experiments