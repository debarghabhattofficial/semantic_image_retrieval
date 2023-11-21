import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

from utils.utils import read_image, add_caption, load_dataset


class ImageCaptioning:
    """
    This class performs image captioning.
    """
    def __init__(self, 
                 model_name, 
                 model_type,
                 input_size,
                 is_eval,
                 batch_size,
                 device,
                 verbose):
        """
        This method initializes the class.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.input_size = input_size
        self.is_eval = is_eval
        self.batch_size = batch_size
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

    def inference_on_batch_of_images(self, 
                                     dataset_path,
                                     out_dir):
        """
        This method performs inference on a batch 
        of images.
        """
        # Load the dataset.
        dataset = load_dataset(
            dataset_path=dataset_path, 
            batch_size=self.batch_size,
            input_size=self.input_size
        )
        # Create a DataLoader to efficiently load batches 
        # of images.
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        # Iterate over batches of data.
        for batch_num, (batch_imgs, batch_lbls) in enumerate(tqdm(data_loader, unit="batch")):
            # Generate caption for the batch of images.
            batch_imgs = batch_imgs.to(self.device)
            output = self.model.generate({"image": batch_imgs})

            # Iterate over individual images in the batch.
            for lbl_num in range(len(batch_lbls)):
                # Get the index of the image in the dataset.
                img_idx = (batch_num * data_loader.batch_size) + lbl_num

                # Extract the (file)name of the image.
                img_path = dataset.imgs[img_idx][0]

                # Generate the parent directory path for
                # saving modified image with generated 
                # caption.
                out_par_dir = os.path.join(
                    out_dir, 
                    dataset.classes[batch_lbls[lbl_num]]
                )

                # Extract the caption from the output and
                # add to the bottom of the image.
                add_caption(
                    img_path=img_path, 
                    caption_text=output[lbl_num],
                    out_dir=out_par_dir
                )

        pass