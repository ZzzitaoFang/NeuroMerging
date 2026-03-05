#!/bin/bash

exp_name=llama2_top10
TOPK=top10
# lambda_code=linear+2.0+2.91+0.2
lambda_code=mergelist+2.7,2.9

CUDA_VISIBLE_DEVICES=0 \
/usr/bin/time -v \
python inference.py \
    --exp_name ${exp_name} \
    --weight_format delta_weight \
    --topk ${TOPK} \
    --lambda_code ${lambda_code}