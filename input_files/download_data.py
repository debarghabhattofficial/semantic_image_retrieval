import os
import sys
import argparse
from pprint import pprint

import boto3


VALID_IMG_EXTS = ["jpg", "jpeg", "png"]
FOLDER_PREFIXES = [
    "camera-clarity-dataset",
    "row_runner/dataset/validation/",
    "object-detection-dataset",
    "row_runner_2/dataset"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource",
        type=str,
        default="s3",
        help="Resource to download from. Default: s3."
    )
    parser.add_argument(
        "--bucket_name", 
        type=str, 
        default="sg-fleet-usage", 
        help="Name of the S3 bucket to download from."
    )
    parser.add_argument(
        "--folder_prefix", 
        type=str, 
        default="camera-clarity-dataset", 
        help="Prefix of the folder to download from."
    )
    parser.add_argument(
        "--local_directory", 
        type=str, 
        help="Local directory to download files to."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="use to print intermediate output for " + \
          "debugging purpose"
    )
    args = parser.parse_args()
    return args


def download_from_resource(resource, 
                           bucket_name, 
                           folder_prefix, 
                           local_directory, 
                           verbose=False):
    resource = boto3.client(resource)

    try: 
        # List all objects in the specified S3 bucket with 
        # the given prefix (folder)
        objects = resource.list_objects_v2(
            Bucket=bucket_name, 
            Prefix=folder_prefix
        )

        # Download every image from the remote S3 bucket 
        # to the local directory.
        if verbose:
            print("Image files: ")
        for obj in objects.get('Contents', []):
            # Extract the object key, i.e., the file name.
            obj_key = obj['Key']
            obj_ext = obj_key.split(".")[-1]
            if obj_ext in VALID_IMG_EXTS:
                # Generate local file path by joining local 
                # directory with object key.
                local_file_path = os.path.join(
                    local_directory, 
                    os.path.abspath(obj_key)
                )
                par_dir = "/".join(local_file_path.split("/")[:-1])
                if not os.path.isdir(par_dir):
                    os.makedirs(par_dir)

                # Download the object from resource to the local file path.
                resource.download_file(
                    Bucket=bucket_name, 
                    Key=obj_key, 
                    Filename=local_file_path
                )
                
                if verbose:
                    print(f"Downloaded {obj_key} to {local_file_path}.")
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        return True


def main():
    args = parse_args()

    result = download_from_resource(
        resource=args.resource,
        bucket_name=args.bucket_name,
        folder_prefix=args.folder_prefix,
        local_directory=args.local_directory,
        verbose=args.verbose
    )
    if result:
        print(f"Downloaded files to {args.local_directory}.")
    else:
        print(f"Could not download files.")

    return


if __name__=="__main__":
    main()