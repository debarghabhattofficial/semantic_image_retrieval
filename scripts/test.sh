#!/bin/bash

cd "../src"

python test.py \
    --cfg_path "../cfg/feat_extract-blip.yaml" \
    --dataset_path "../input_images/auto_image_labelling/bushy-patchy" \
    --img_path "../input_images/test_sample_1.jpg" \
    --text_input "bushy" "patchy" \
    --out_dir "../results/auto_image_labelling/blip/bushy-patchy" \
    --plot_dir "../plots/auto_image_labelling/blip/bushy-patchy" \
    --infer_batch \
    --project_lower \
    --compute_centroids \
    --vis_pca \
    --save_plots \
    --save_embeds \
    --compute_similarity \
    --save_sim_scores \
    --verbose