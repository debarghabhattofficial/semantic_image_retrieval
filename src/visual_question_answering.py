import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

from utils.utils import read_image, add_caption, load_dataset


class PromptTemplate:
    """
    This class defines the template of the input
    prompts of the VQA model.
    """
    def __init__(self, template):
        self.template = template

    def generate_prompt(self, *args):
        """
        This method generates the prompt from the
        template and the arguments.
        """
        formatted_prompt = None
        try:
            formatted_prompt = self.template.format(*args)
        except Exception as e:
            err_msg = f"Error: {e}\n" + \
                "Ensure no. of placeholders matches no. of arguments."
            print(err_msg)
            formatted_prompt =  None

        return formatted_prompt


class VisualQuestionAnswering:
    """
    This class performs visual question answering.
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

    def get_formatted_prompt(self,
                             qa_pair):
        """
        This method returns a formated prompt.
        """
        formatted_prompt = f"Question: {qa_pair[0]} " + \
            f"Answer: {qa_pair[1]}"
        return formatted_prompt


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

            context_history = []
            # Generate prompts and asks questiosn to
            # the model.
            while True:
                cur_ques = input("Enter question: ")
                cur_ques = str(cur_ques)
                cur_prompt = " ".join(
                    list(map(
                        lambda context: self.get_formatted_prompt(context), 
                        context_history
                    ))
                ) + self.get_formatted_prompt((cur_ques, ""))
                print(f"Prompt: {cur_prompt}")
                # output = self.model.generate(
                #     {"image": img_tensor, "prompt": cur_prompt}
                # )
                cnt_msg = "Press 'N' to abort, " + \
                    "or any other key to ask another question."
                cnt = input(cnt_msg)
                cnt = str(cnt)
                if (cnt == "N") or (cnt == "n"):
                    break

            # Extract the caption from the output and
            # add to the bottom of the image.
            # add_caption(
            #     img_path=img_path, 
            #     caption_text=output[0],
            #     out_dir=out_dir
            # )
        else:
            print("Image not found.")

        return

    def inference_on_batch_of_images(self, 
                                     dataset_path,
                                     out_dir):
        """
        This method performs inference on a batch 
        of images.
        TODO: Implement this method later. This might
        not be even required for the task of vqa where
        we generally focus on a single image.
        """
        pass