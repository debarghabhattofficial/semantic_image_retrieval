import os
import sys
import argparse
from tqdm import tqdm
from pprint import pprint

import boto3


VALID_IMG_EXTS = ["jpg", "jpeg", "png"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource",
        type=str,
        default="s3",
        help="Resource to upload to. Default: s3."
    )
    parser.add_argument(
        "--bucket_name", 
        type=str, 
        default="sg-fleet-usage", 
        help="Name of the S3 bucket to upload to."
    )
    parser.add_argument(
        "--folder_prefix", 
        type=str, 
        default="image_captioning", 
        help="Prefix of the folder to upload to."
    )
    parser.add_argument(
        "--local_path",
        type=str, 
        help="Path to the data to be uploaded."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Use to print intermediate output for " + \
          "debugging purpose."
    )
    args = parser.parse_args()
    return args


def upload_to_resource(resource, 
                       bucket_name, 
                       folder_prefix, 
                       local_path,
                       verbose=False):
    """
    This method uploads a local file or directory to a
    resource (e.g., S3) at a given path.
    """
    file_upload_success = True

    resource = boto3.client(resource)
    try:
        # Walk through each file in the local path.
        for root, dirs, files in os.walk(local_path):
            for file in files:
                # Get the local path of the file.
                local_file_path = os.path.join(root, file)
                # print(f"local_file_path: {local_file_path}")
                # print("-" * 75)
                relative_path = os.path.relpath(local_file_path, local_path)
                # print(f"relative_path: {relative_path}")
                # print("-" * 75)

                # Generate the remote path.
                resource_path = os.path.join(folder_prefix, relative_path) if folder_prefix else relative_path
                # print(f"resource_path: {resource_path}")
                # print("-" * 75)

                # 4. Upload each file to the resource if 
                # it's an image.
                if file.split(".")[-1] in VALID_IMG_EXTS:
                    resource.upload_file(
                        Bucket=bucket_name, 
                        Key=resource_path,
                        Filename=local_file_path, 
                    )
    except Exception as e:
        print(f"Error: {e}")
        file_upload_success = False
    finally:
        file_upload_success = True

    return file_upload_success


def main():
    args = parse_args()

    result = upload_to_resource(
        resource=args.resource, 
        bucket_name=args.bucket_name, 
        folder_prefix=args.folder_prefix, 
        local_path=args.local_path
    )
    if result:
        remote_dir = os.path.join(
            args.bucket_name, 
            args.folder_prefix
        )
        print(f"Uploaded files to {remote_dir}.")
    else:
        print(f"Could not upload files.")
    return


if __name__=="__main__":
    main()