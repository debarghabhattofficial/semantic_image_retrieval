#!/bin/bash

# Available images directories in AWS S3 bucket:
# "camera-clarity-dataset"
# "row_runner/dataset/validation/"
# "object-detection-dataset"
# "row_runner_2/dataset"

# Execute the python script.
python download_data.py \
    --resource "s3" \
    --bucket_name "sg-fleet-usage" \
    --folder_prefix "camera-clarity-dataset" \
    --local_directory "/media/dbhattacharjee/Deb/projects/semantic_image_retrieval/input_files"