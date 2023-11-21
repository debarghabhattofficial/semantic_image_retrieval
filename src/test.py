import argparse
import yaml

import torch

from image_captioning import read_image, add_caption
from image_captioning import ImageCaptioning


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to config file."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="Path to input image."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to output directory to store results."
    )
    parser.add_argument(
        "--process_batch", 
        action="store_true",
        help="Use to process batch input."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Use to print intermediate output for " + \
          "debugging purpose."
    )

    args = parser.parse_args()
    return args


def load_config(config_path):
    """
    This method reads the configuration from a
    YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    args = parse_args()
    config = load_config(args.cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_cap = ImageCaptioning(
        model_name=config["model"]["name"],  # "blip2_t5", 
        model_type=config["model"]["type"],  # "caption_coco_flant5xl", 
        input_size=tuple(config["model"]["input_size"]),
        is_eval=config["model"]["is_eval"],
        device=device,
        verbose=args.verbose
    )

    if not args.process_batch:
        img_cap.inference_on_single_image(
            img_path=args.img_path,
            out_dir=args.out_dir
        )
    else:
        # TODO: Implement batch inference.
        pass

    return


if __name__ == "__main__":
    main()