import os
import argparse

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_path",
        type=str,
        help="Path to input image."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="use to print intermediate output for " + \
          "debugging purpose"
    )

    args = parser.parse_args()
    return args


def read_image(img_path):
    # Read the image.
    image = Image.open(img_path)

    # Define a transform to convert the image to tensor.
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(image)

    # Print the converted image tensor
    return tensor


def main():
    args = parse_args()
    img_path = args.img_path
    verbose = args.verbose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", 
        model_type="base_coco", 
        is_eval=True, 
        device=device
    )

    # Preprocess the image and return tensor.
    image = None
    if os.path.exists(img_path):
        print("Image found.")
        image = read_image(img_path).to(device)
        if verbose:
            print(f"image.shape: {image_shape}")
            print(f"image: \n{image}")
            print("-" * 75)
    else:
        print("Image not found.")

    # Generate caption for the image.
    model.generate({"image": image})
    print(f"model keys: {model.keys()}")
    print("-" * 75)


if __name__ == "__main__":
    main()