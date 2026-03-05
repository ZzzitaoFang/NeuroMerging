#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
/usr/bin/time -v python merge_7b.py \
    --merge_chinese \
    --merge_math \
    --merge_code \
    --merging_method_name  neuro_merging \
    --param_value_mask_rate 0.95