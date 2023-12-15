import os
import pickle

from torchvision import transforms, datasets

from PIL import Image, ImageDraw, ImageFont
import textwrap


def read_images(unit="single",
                img_path=None,
                batch_size=None):
    """
    This method:
        1. Reads an image from the given path if unit is "single".
        2. Creates a custom dataset from a give directory if unit is "batch".
    """
    img_pil = None
    img_dataset = None
    if unit == "single":
        # Read the image.
        img_pil = Image.open(img_path)
    elif unit == "batch":
        # Define a transform to convert to PIL image.
        transform = transforms.Compose([  
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
        ])
        # Create a custom dataset of images.
        img_dataset = datasets.ImageFolder(
            root=img_path,
            transform=transform
        )

    # Return image or dataset.
    return img_pil if unit == "single" else img_dataset


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


def save_pickle_data(data, file_name, directory, label=None):
    "This method saves the data as a pickle file."
    # Create output directory (if doesn't exist).
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save the data as a pickle with given file_name.
    path = os.path.join(directory, file_name)
    with open(path, "wb") as file:
        # A new file will be created
        pickle.dump(data, file)

    if label is None:
        print(f"Saved data as a pickle file at: {path}")
    else:
        print(f"Saved {label} as a pickle file at: {path}")

    return


def load_pickle_data(path, label=None):
    "This method loads pickled data from given path."
    data = None

    # Load pickled data from given path (if it exists).
    if os.path.exists(path):
        with open(path, "rb") as file:
            data = pickle.load(file)

        if label is None:
            print(f"Loaded pickled data from: {path}")
        else:
            print(f"Loaded pickled {label} from: {path}")

    return data