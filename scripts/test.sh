#!/bin/bash

cd "../src"

python test.py \
    --cfg_path "../cfg/vqa-blip.yaml" \
    --dataset_path "../input_images/test_dataset" \
    --img_path "../input_images/test_sample_1.jpg" \
    --out_dir "../results/vqa/blip/" \
    --verbose