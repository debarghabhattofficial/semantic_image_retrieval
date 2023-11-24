#!/bin/bash

# Available images directories in AWS S3 bucket:
# "camera-clarity-dataset"
# "row_runner/dataset/validation/"
# "object-detection-dataset"
# "row_runner_2/dataset"

# Execute the python script.
python upload_data.py \
    --resource "s3" \
    --bucket_name "sg-fleet-usage" \
    --folder_prefix "image_captioning" \
    --local_path "../results/image_captioning"