#!/bin/bash

cd "../src"

python test.py \
    --cfg_path "../cfg/image_captioning.yaml" \
    --img_path "../input_files/test_sample_1.jpg" \
    --out_dir "../results/image_captioning/blip" \
    --verbose