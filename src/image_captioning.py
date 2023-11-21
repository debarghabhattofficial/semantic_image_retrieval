import os
import argparse

import torch
from lavis.models import load_model_and_preprocess

from utils.utils import read_image, add_caption


class ImageCaptioning:
    """
    This class performs image captioning.
    """
    def __init__(self, 
                 model_name, 
                 model_type,
                 input_size,
                 is_eval,
                 device,
                 verbose):
        """
        This method initializes the class.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.is_eval = is_eval
        self.input_size = input_size
        self.device = device

        # Load the model and image processors.
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=self.model_name, 
            model_type=self.model_type,
            is_eval=self.is_eval,
            device=self.device
        )

        self.verbose = verbose

        return


    def inference_on_single_image(self, 
                                  img_path,
                                  out_dir):
        """
        This method performs inference on a single image.
        """
        image = None
        if os.path.exists(img_path):
            # Preprocess the image and return tensor.
            img_tensor = read_image(img_path, self.input_size).to(self.device)
            if self.verbose:
                print(f"img_tensor.shape: {img_tensor.shape}")
                print(f"img_tensor: \n{img_tensor}")
                print("-" * 75)

            # Generate caption for the image.
            output = self.model.generate({"image": img_tensor})
            print(f"output: {output}")
            print("-" * 75)

            # Extract the caption from the output and
            # add to the bottom of the image.
            add_caption(
                img_path=img_path, 
                caption_text=output[0],
                out_dir=out_dir
            )
        else:
            print("Image not found.")

        return