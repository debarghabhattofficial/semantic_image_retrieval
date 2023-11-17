#!/bin/bash

# Execute the python script.
python download_data.py \
    --resource "s3" \
    --bucket_name "sg-fleet-usage" \
    --folder_prefix "object-detection-dataset" \
    --local_directory "/media/dbhattacharjee/Deb/projects/semantic_image_retrieval/input_files"