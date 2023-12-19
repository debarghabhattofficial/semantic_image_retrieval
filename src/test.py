import argparse
import yaml

import torch

from feature_extractor import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to config file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to dataset containing test images."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="",
        help="Path to input image."
    )
    parser.add_argument(
        "--text_input",
        type=str,
        default="",
        help="Input text for the model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to output directory to store feature embeddings."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        help="Path to output directory to store output plots (if any)."
    )
    parser.add_argument(
        "--infer_batch", 
        action="store_true",
        help="Use to run inference on batch input."
    )
    parser.add_argument(
        "--compute_centroids", 
        action="store_true",
        help="Use to compute class centroids in case of " + \
            "multiple inputs."
    )
    parser.add_argument(
        "--project_lower", 
        action="store_true",
        help="Use if you want to work with normalized " + \
            "low-dimensional features."
    )
    parser.add_argument(
        "--vis_pca", 
        action="store_true",
        help="Use to visualise data points after " + \
            "applying PCA decomposition."
    )
    parser.add_argument(
        "--save_embeds", 
        action="store_true",
        help="Use to save feature emebeddings of individual " + \
            "input or class-wise feature embedding centroids " + \
            "in case of multiple inputs."
    )
    parser.add_argument(
        "--save_plots", 
        action="store_true",
        help="Use to save 2D plots of feature embeddings space " + \
            "with different class-specific data points after " + \
            "dimensionality reduction using PCA."
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

    feature_extractor = FeatureExtractor(
        model_name=config["model"]["name"],
        model_type=config["model"]["type"],
        input_size=tuple(config["model"]["input_size"]),
        is_eval=config["model"]["is_eval"],
        batch_size=config["batch_size"],
        device=device,
        verbose=args.verbose
    )

    if not args.infer_batch:
        feature_extractor.infer_single_input(
            img_path=args.img_path,
            project_lower=args.project_lower,
            in_text=args.text_input,
            out_dir=args.out_dir,
            save_embeds=args.save_embeds
        )
    else:
        feature_extractor.infer_batch_of_inputs(
            img_path=args.dataset_path,
            project_lower=args.project_lower,
            compute_centroids=args.compute_centroids,
            out_dir=args.out_dir,
            plot_dir=args.plot_dir,
            vis_pca=args.vis_pca,
            save_embeds=args.save_embeds,
            save_plots=args.save_plots
        )

    return


if __name__ == "__main__":
    main()