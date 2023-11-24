import os

from torchvision import transforms, datasets

from PIL import Image, ImageDraw, ImageFont
import textwrap


def read_image(img_path, input_size=(384, 384)):
    # Read the image.
    img_pil = Image.open(img_path)

    # Define a transform to convert the image to tensor.
    # TODO: Input image size for BLIP model is 384 x 384.
    # Verify after reading about it from the research paper.
    transform = transforms.Compose([
        transforms.Resize(input_size),  
        transforms.ToTensor(),
    ])

    # Convert the image to PyTorch tensor.
    img_tensor = transform(img_pil).unsqueeze(0)

    return img_tensor


def load_dataset(dataset_path, 
                 batch_size,
                 input_size=(384, 384)):
    # Define the transformation to be applied to each image
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize the image.
        transforms.ToTensor(),  # Convert the image to a tensor.
    ])

    # Create a custom dataset.
    custom_dataset = datasets.ImageFolder(
        root=dataset_path, 
        transform=transform
    )

    return custom_dataset


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

    # get draw object to add a caption at the 
    # bottom of the canvas.
    draw = ImageDraw.Draw(canvas)

    #  Specify font type and color.
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 
            size=20
        )
    except IOError:
        font = ImageFont.load_default()
    caption_font_color = (0, 0, 0)  # Black

    # Wrap caption text to allow maximum of 40 
    # characters in a single line.
    caption_text = textwrap.wrap(
        caption_text, 
        width=40
    )

    # Add caption text line-by-line.
    for i, line in enumerate(caption_text):
        line_position = (
            10, 
            img_pil.height + 10 + (i * 20)
        )
        draw.text(
            line_position, 
            line, 
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