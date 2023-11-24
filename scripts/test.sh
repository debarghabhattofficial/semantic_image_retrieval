#!/bin/bash

cd "../src"

python test.py \
    --cfg_path "../cfg/image_captioning.yaml" \
    --dataset_path "../input_images/test_dataset" \
    --img_path "../input_images/test_sample_1.jpg" \
    --out_dir "../results/image_captioning/blip/test_dataset" \
    --infer_batch \
    --verbose