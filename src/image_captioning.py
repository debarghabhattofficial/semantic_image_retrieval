import os
import argparse

import torch
import torchvision.transforms as transforms
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--verbose", 
        action="store_true",
        help="use to print intermediate output for " + \
          "debugging purpose"
    )

    args = parser.parse_args()
    return args


def read_image(img_path):
    # Read the image.
    img_pil = Image.open(img_path)

    # Define a transform to convert the image to tensor.
    # TODO: Input image size for BLIP model is 384 x 384.
    # Verify after reading about it from the research paper.
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  
        transforms.ToTensor(),
    ])

    # Convert the image to PyTorch tensor.
    img_tensor = transform(img_pil).unsqueeze(0)

    return img_tensor


def add_caption(img_path, 
                caption_text, 
                out_dir):
    """
    This method adds a caption at the bottom of
    an image.
    """
    # # Step 1: Read input image.
    img_pil = Image.open(img_path)

    # Create a rectangular canvas with off-white 
    # background color.
    canvas = Image.new(
        "RGB", 
        (img_pil.width, img_pil.height + 100), 
        (245, 245, 245)
    )

    # Copy the input image on the top portion of 
    # the canvas.
    canvas.paste(img_pil, (0, 0))

    # Add a caption at the bottom of the canvas.
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 
            size=20
        )
    except IOError:
        font = ImageFont.load_default()
    caption_position = (10, img_pil.height + 10)
    caption_font_color = (0, 0, 0)  # Black
    draw.text(
        caption_position, 
        caption_text, 
        font=font, 
        fill=caption_font_color
    )

    # Save the modified image having caption.
    if not out_dir is None:
        # Create directory is it does not exist.
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # Set the output path.
        out_path = os.path.join(
            out_dir, os.path.basename(img_path)
        )
        # Save the modified image having caption.
        canvas.save(out_path)

    return


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads BLIP/BLIP2 caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",  # "blip2_t5", 
        model_type="base_coco",  # "caption_coco_flant5xl", 
        is_eval=True, 
        device=device
    )

    image = None
    if os.path.exists(args.img_path):
        print("Image found.")
        # Preprocess the image and return tensor.
        img_tensor = read_image(args.img_path).to(device)
        if args.verbose:
            print(f"img_tensor.shape: {img_tensor.shape}")
            print(f"img_tensor: \n{img_tensor}")
            print("-" * 75)

        # Generate caption for the image.
        output = model.generate({"image": img_tensor})
        print(f"output: {output}")
        print("-" * 75)

        # Extract the caption from the output and
        # add to the bottom of the image.
        add_caption(
            img_path=args.img_path, 
            caption_text=output[0],
            out_dir=args.out_dir
        )
    else:
        print("Image not found.")

    return


if __name__ == "__main__":
    main()