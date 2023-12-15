import os
import argparse
from tqdm import tqdm
from pprint import pprint

import numpy as np
from PIL import Image as PilImage

import torch
from torch_scatter import scatter_sum, scatter_mean
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchsummary import summary

from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

from utils.utils import read_images, load_pickle_data, save_pickle_data


class FeatureExtractor:
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
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=self.model_name, 
            model_type=self.model_type,
            is_eval=self.is_eval,
            device=self.device
        )
        self.verbose = verbose

        return

    def preprocess_single_input(self, in_img, in_text):
        """
        This method preprocesses the input image and text.
        """
        # Preprocess input image and return tensor.
        in_img = self.vis_processors["eval"](
            in_img
        ).unsqueeze(0).to(self.device)

        # Preprocess input text and return tensor.
        in_text = self.txt_processors["eval"](in_text)

        return in_img, in_text

    def preprocess_batch_of_inputs(self, img_batch, text_batch):
        """
        This method preprocesses the input image batch 
        and input text batch.
        """
        to_pil_transform = ToPILImage()
        # Preprocess input image and return tensor.
        img_batch = torch.stack([
            self.vis_processors["eval"](to_pil_transform(in_img))
            for in_img in img_batch
        ]).to(self.device)

        # Preprocess input text and return tensor.
        text_batch = [
            self.txt_processors["eval"](in_text)
            for in_text in text_batch
        ]

        return img_batch, text_batch

    def extract_from_single_input(self, 
                                  img_path,
                                  in_text,
                                  out_dir):
        """
        This method extract featurws from a single input.
        """
        image = None
        if os.path.exists(img_path):
            # Read image from given path.
            in_img = read_images(
                unit="single",
                img_path=img_path,
            )
            # Preprocess read image and text.
            in_img, in_text = self.preprocess_single_input(
                in_img=in_img,
                in_text=in_text
            )
            sample = {"image": in_img, "text_input": in_text}
            print("sample: ")
            pprint(sample)
            print("-" * 75)

            # Extract image features.
            img_feats = self.model.extract_features(sample, mode="image")
            print(f"img_feats.image_embeds shape: {img_feats.image_embeds.shape}")
            print(f"img_feats.image_embeds): \n{img_feats.image_embeds}")
            print("-" * 75)

            if False:
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

    def update_centroid(self, 
                        prev_centroid, 
                        prev_counts, 
                        new_centroid, 
                        new_counts):
        """
        This method updates the centroid of a cluster.
        """
        updated_counts = prev_counts + new_counts
        updated_centroid = ((prev_counts * prev_centroid) + (new_counts * new_centroid)) / (prev_counts + new_counts)
        return (updated_counts, updated_centroid)

    def update_class_stats(self,
                           class_stats,
                           batch_stats):
        """
        This method updates the class-level statistics.
        """
        max_lbl = len(batch_stats)

        for i in range(max_lbl):
            if i in class_stats:
                class_stats[i] = self.update_centroid(
                    prev_centroid=class_stats[i][1],
                    prev_counts=class_stats[i][0],
                    new_centroid=batch_stats[i][1],
                    new_counts=batch_stats[i][0]
                )
            else:
                class_stats[i] = batch_stats[i]

        return class_stats

    def extract_from_batch_of_inputs(self, 
                                     img_path,
                                     out_dir):
        """
        This method performs inference on a batch 
        of images.
        """
        # Load the dataset.
        dataset = read_images(
            unit="batch",
            img_path=img_path, 
            batch_size=self.batch_size
        )
        # Create a DataLoader to efficiently load batches 
        # of images.
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        # Dictionary to store class-level statistics.
        class_stats = {}  # Keys: Class labels, Values: (Label counts, Centroids).
        # Iterate over batches of data.
        for batch_num, (batch_imgs, batch_lbls) in enumerate(tqdm(data_loader, unit="batch")):
            print(f"class_stats (BEFORE): ")
            pprint(class_stats)
            print("-" * 75)

            batch_lbls = batch_lbls.to(self.device)

            # Preprocess image and text inputs.
            batch_texts = [""] * self.batch_size
            batch_imgs, batch_txts = self.preprocess_batch_of_inputs(
                img_batch=batch_imgs,
                text_batch=batch_texts,
            )
            sample = {"image": batch_imgs, "text_input": batch_texts}

            # Extract image features.
            img_feats = self.model.extract_features(sample, mode="image")

            # Compute batch-level statistics.
            # Compute class label counts for current batch.
            lbl_counts = scatter_sum(
                src=torch.ones_like(batch_lbls),
                index=batch_lbls,
                dim=0
            )
            # Compute class centroids for current batch.
            batch_centroid = scatter_mean(
                src=img_feats.image_embeds, 
                index=batch_lbls, 
                dim=0
            )
            # Combine to form batch-level statistics.
            batch_stats = list(zip(
                lbl_counts.cpu().numpy(), 
                batch_centroid.cpu().numpy()
            ))

            # Update class-level statistics.
            class_stats = self.update_class_stats(
                class_stats=class_stats,
                batch_stats=batch_stats
            )
            print(f"class_stats (AFTER): ")
            pprint(class_stats)
            print("=" * 75)

        # Update class-level statistics's keys to respective 
        # class labels for saving as pickle file.
        class_stats = {
            dataset.classes[k]: v
            for k, v in class_stats.items()
        }
        print(f"class_stats: ")
        pprint(class_stats)
        print("-" * 75)

        # Save class-level statistics as pickle file.
        if False:
            save_pickle_file(
                data=class_stats,
                file_name="class_stats.pkl",
                directory=out_dir,
                label="class-level statistics"
            )

        return