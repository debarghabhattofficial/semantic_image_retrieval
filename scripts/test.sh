#!/bin/bash

cd "../src"

python test.py \
    --cfg_path "../cfg/vqa-blip.yaml" \
    --dataset_path "../input_images/test_dataset" \
    --img_path "../input_images/row_runner_2_vines_april_block1_0102.jpg" \
    --out_dir "../results/vqa/blip/" \
    --verbose