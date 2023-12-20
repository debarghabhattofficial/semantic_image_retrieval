import os
import argparse
from tqdm import tqdm
from pprint import pprint

import numpy as np
from PIL import Image as PilImage
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import torch
from torch_scatter import scatter_sum, scatter_mean
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from lavis.models import load_model_and_preprocess

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

    def visualise_using_pca(self,
                            feat_embeds,
                            class_labels,
                            class_stats=None,
                            plot_dir=None,
                            save_plots=False):
        """
        This method visualises the feature embeddings
        using PCA.
        """

        # Flatten the 2D feature embeddings.
        feat_embeds = feat_embeds.flatten(start_dim=1)

        # Use PCA to reduce the dimensionality of the
        # feature embeddings.
        pca_reduce = PCA(n_components=2)
        feat_embeds = pca_reduce.fit_transform(feat_embeds)

        df = pd.DataFrame()
        df["label"] = class_labels
        df["x1"] = feat_embeds[:, 0]
        df["x2"] = feat_embeds[:, 1]

        # Visualize the feature embeddings.
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(w=8, h=8)

        sns.scatterplot(
            data=df, 
            x="x1", 
            y="x2", 
            hue="label", 
            palette=sns.color_palette("muted"),
            legend="full"
        )
        
        if class_stats is not None:
            # Visualize the class centroids.
            tab10_colors = plt.cm.tab10.colors
            tab10_cmap = ListedColormap(tab10_colors)
            for k, (lbl, (lbl_counts, lbl_centroid)) in enumerate(class_stats.items()):
                lbl_centroid = pca_reduce.transform(
                    lbl_centroid.unsqueeze(0).flatten(start_dim=1).cpu()
                )
                plt.scatter(
                    x=lbl_centroid[:, 0],
                    y=lbl_centroid[:, 1],
                    c=[tab10_cmap(k)],
                    marker="^",
                    s=100,
                    label=lbl
                )
                plt.annotate(
                    text=lbl,
                    xy=(lbl_centroid[:, 0], lbl_centroid[:, 1]),
                    xytext=(lbl_centroid[:, 0] + 0.1, lbl_centroid[:, 1] + 0.1),
                )

        plt.tight_layout()
        if save_plots and (plot_dir is not None):
            plt_dir, plt_name = os.path.split(plot_dir)
            if os.path.isdir(plt_dir) == False:
                os.makedirs(plt_dir, exist_ok=True)
            plt_file_name = os.path.join(
                plt_dir, f"{plt_name}.png"
            )
            plt.savefig(plt_file_name)
        plt.show()
        plt.close()

        return

    def preprocess_batch_of_inputs(self, img_batch=None, text_batch=None):
        """
        This method preprocesses the input image batch 
        and input text batch.
        """
        # Preprocess input image and return tensor.
        if img_batch is not None:
            to_pil_transform = ToPILImage()
            img_batch = torch.stack([
                self.vis_processors["eval"](to_pil_transform(in_img))
                for in_img in img_batch
            ]).to(self.device)

        # Preprocess input text and return tensor.
        if text_batch is not None:
            text_batch = [
                self.txt_processors["eval"](in_text)
                for in_text in text_batch
            ]

        return img_batch, text_batch

    def infer_single_input(self, 
                           img_path,
                           in_text="",
                           project_lower=False,
                           out_dir=None,
                           save_embeds=False):
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

            # Extract image features.
            img_feats = self.model.extract_features(sample, mode="image")
            if project_lower:
                img_feats = img_feats.image_embeds_proj
            else:
                img_feats = img_feats.image_embeds

            # Save class-level statistics as pickle file.
            if (save_embeds == True) and (out_dir is not None):
                save_pickle_data(
                    data=img_feats.cpu().numpy(),
                    file_name="image_embeds.pkl",
                    directory=out_dir,
                    label="image embeddings"
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
        updated_centroid = (
            (prev_counts * prev_centroid) + (new_counts * new_centroid)
        ) / updated_counts
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

    def infer_batch_of_inputs(self, 
                              img_path,
                              in_text=None,
                              project_lower=False,
                              compute_centroids=False,
                              out_dir=None,
                              plot_dir=None,
                              vis_pca=False,
                              compute_similarity=False,
                              save_embeds=False,
                              save_plots=False,
                              save_sim_scores=False,
                              save_class_report=False):
        """
        This method performs inference on a batch 
        of images.
        """
        # Sort the input text labels in ascending order.
        in_text.sort()
        
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
        
        # List to store other data point-level statistics.
        data_points_labels = []
        if vis_pca:
            data_points_embeds = []
        if compute_similarity:
            data_points_sim_scores = []
            data_points_class_probs = []

        # Iterate over batches of data.
        for batch_num, (batch_imgs, batch_lbls) in enumerate(tqdm(data_loader, unit="batch")):
            batch_lbls = batch_lbls.to(self.device)

            # Preprocess image and text inputs.
            batch_texts = [""] * self.batch_size
            batch_imgs, batch_txts = self.preprocess_batch_of_inputs(
                img_batch=batch_imgs,
                text_batch=batch_texts,
            )
            sample = {"image": batch_imgs, "text_input": batch_texts}

            # Extract image features.
            img_feats = None
            if "clip" in self.model_name:
                with torch.no_grad():
                    img_feats = self.model.extract_features(sample)
            else:
                img_feats = self.model.extract_features(sample, mode="image")
            if project_lower:
                img_feats = img_feats.image_embeds_proj
            else:
                img_feats = img_feats.image_embeds
            
            data_points_labels.extend(batch_lbls.cpu())

            if vis_pca:
                data_points_embeds.extend(img_feats.cpu())

            if compute_centroids:
                # Compute batch-level statistics (class centroids, class counts).
                # Compute class label counts for current batch.
                lbl_counts = scatter_sum(
                    src=torch.ones_like(batch_lbls),
                    index=batch_lbls,
                    dim=0
                )
                # Compute class centroids for current batch.
                batch_centroid = scatter_mean(
                    src=img_feats, 
                    index=batch_lbls, 
                    dim=0
                )
                # Combine to form batch-level statistics.
                batch_stats = list(zip(
                    lbl_counts.cpu(),
                    batch_centroid.cpu()
                ))
                # Update class-level statistics.
                class_stats = self.update_class_stats(
                    class_stats=class_stats,
                    batch_stats=batch_stats
                )

            text_feats = None
            if compute_similarity and (in_text is not None):
                batch_imgs = batch_imgs.repeat_interleave(
                    len(in_text), dim=0
                )
                img_feats = img_feats.repeat_interleave(
                    len(in_text), dim=0
                )

                batch_texts = in_text * min(self.batch_size, batch_lbls.shape[0])
                _, batch_txts = self.preprocess_batch_of_inputs(
                    text_batch=batch_texts,
                )
                sample = {"image": batch_imgs, "text_input": batch_texts}

                # Extract text features.
                if "clip" in self.model_name:
                    with torch.no_grad():
                        text_feats = self.model.extract_features(sample)
                else:
                    text_feats = self.model.extract_features(sample, mode="text")
                if project_lower:
                    text_feats = text_feats.text_embeds_proj
                else:
                    text_feats = text_feats.text_embeds
                
                # Compute similarity between image and text features.
                sim_scores = torch.diag(
                    img_feats[:, 0, :] @ text_feats[:, 0, :].t()
                ).reshape(-1, len(in_text))
                class_probs = torch.softmax(sim_scores / 0.01, dim=1)

                data_points_sim_scores.extend(sim_scores.cpu())
                data_points_class_probs.extend(class_probs.cpu())

        if compute_centroids:
            # Update class-level statistics's keys to respective 
            # class labels for saving as pickle file.
            class_stats = {
                dataset.classes[k]: v
                for k, v in class_stats.items()
            }

            # Save class-level statistics as pickle file.
            if (save_embeds == True) and (out_dir is not None):
                save_pickle_data(
                    data=class_stats,
                    file_name="class_stats.pkl",
                    directory=out_dir,
                    label="class-level statistics"
                )

        data_points_labels = torch.stack(
            data_points_labels, dim=0
        ).cpu().numpy()

        if vis_pca:
            data_points_embeds = torch.stack(data_points_embeds, dim=0)
            data_points_label_names = [
                dataset.classes[lbl]
                for lbl in data_points_labels
            ]
            self.visualise_using_pca(
                feat_embeds=data_points_embeds,
                class_labels=data_points_label_names,
                class_stats=class_stats,
                plot_dir=plot_dir,
                save_plots=save_plots
            )

         # Save feature embeddings and class labels of 
         # the different data points as pickle file.
        if (save_embeds == True) and (out_dir is not None):
            data_point_stats = {
                "embeds": data_points_embeds.cpu().numpy(),
                "labels": data_points_label_names
            }
            save_pickle_data(
                data=data_point_stats,
                file_name="data_point_stats.pkl",
                directory=out_dir,
                label="data point statistics"
            )

        # Save similarity scores and class probabilities of 
        # the different data points as pickle file.
        if (save_sim_scores == True) and (out_dir is not None):
            # Compute average similarity and probability scores 
            # across all data points.
            data_points_sim_scores = torch.stack(
                data_points_sim_scores, dim=0
            ).cpu().numpy()
            data_points_class_probs = torch.stack(
                data_points_class_probs, dim=0
            ).cpu().numpy()
            err_msg = "Class labels and input text labels mismatch, " + \
                "should be equal to 2."
            assert data_points_sim_scores.shape[1] == len(in_text) == 2, err_msg
            avg_sim_scores = np.zeros(shape=(len(in_text),))
            avg_class_probs = np.zeros(shape=(len(in_text),))
            for i_lbl in range(len(in_text)):
                req_data_points = np.where(data_points_labels == i_lbl)
                req_data_points = req_data_points[0] if req_data_points[0].shape[0] > 0 else []
                # Compute avg. similarity scores.
                avg_sim_scores[i_lbl] = np.mean(
                    data_points_sim_scores[req_data_points, i_lbl]
                )
                # Compute avg. class probabilities.
                avg_class_probs[i_lbl] = np.mean(
                    data_points_class_probs[req_data_points, i_lbl]
                )
            data_point_stats2 = {
                "sim_scores": data_points_sim_scores,
                "class_probs": data_points_class_probs,
                "avg_sim_scores": avg_sim_scores,
                "avg_class_probs": avg_class_probs,
                "labels": dataset.classes
            }
            save_pickle_data(
                data=data_point_stats2,
                file_name="data_point_sim_pro_scores.pkl",
                directory=out_dir,
                label="data point similarity and probability scores"
            )

        # Save class-wise classification report as a pickle file.
        if (save_class_report == True) and (out_dir is not None):
            # Generate classification report.
            y_pred = np.argmax(data_points_class_probs, axis=1)
            print(f"y_pred: \n{y_pred}")
            print(f"y_true: \n{data_points_labels}")
            class_report = classification_report(
                y_true=data_points_labels, 
                y_pred=y_pred,
                target_names=dataset.classes,
                output_dict=True
            )
            print(f"class_report")
            pprint(class_report)
            print("-" * 75)
            # Save class report.
            save_pickle_data(
                data=class_report,
                file_name="classification_report.pkl",
                directory=out_dir,
                label="classification report"
            )
        return