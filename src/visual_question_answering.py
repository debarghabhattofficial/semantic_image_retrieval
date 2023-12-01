import os
import argparse
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

from utils.utils import read_image
from utils.utils import load_dataset, load_question_set
from utils.utils import add_context_history


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
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
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

    def infer_using_blip(self,
                         img_tensor):
        """
        This method runs inference using the BLIP model.
        """
        context_history = []
        # Generate prompts and asks questions to
        # the model.
        while True:
            cur_ques = input("Enter question: ")
            cur_ques = str(cur_ques).strip().lower()
            cur_prompt = " ".join(
                list(map(
                    lambda context: self.get_formatted_prompt(context), 
                    context_history
                ))
            ) + " " + self.get_formatted_prompt((cur_ques, ""))
            print(f"Prompt: {cur_prompt}")
            cur_ans = ""
            output = self.model.predict_answers(
                {"image": img_tensor, "text_input": cur_ques},
                inference_method="generate"
            )
            cur_ans = output[0]
            print(f"Answer: {cur_ans}")
            context_history.append((cur_ques, cur_ans))
            cnt_msg = "Press 'N' to abort, " + \
                "or any other key to ask another question: "
            cnt = input(cnt_msg)
            cnt = str(cnt)
            if (cnt == "N") or (cnt == "n"):
                break
                
        return context_history

    def infer_using_blip2(self,
                          img_tensor):
        """
        This method runs inference using the BLIP-2 model.
        """
        context_history = []
        # Generate prompts and asks questions to
        # the model.
        while True:
            cur_ques = input("Enter question: ")
            cur_ques = str(cur_ques).strip().lower()
            cur_prompt = " ".join(
                list(map(
                    lambda context: self.get_formatted_prompt(context), 
                    context_history
                ))
            ) + " " + self.get_formatted_prompt((cur_ques, ""))
            print(f"Prompt: {cur_prompt}")
            cur_ans = ""
            output = self.model.generate(
                {"image": img_tensor, "prompt": cur_prompt},
            )
            cur_ans = output[0]
            print(f"Answer: {cur_ans}")
            context_history.append((cur_ques, cur_ans))
            cnt_msg = "Press 'N' to abort, " + \
                "or any other key to ask another question: "
            cnt = input(cnt_msg)
            cnt = str(cnt)
            if (cnt == "N") or (cnt == "n"):
                break
                
        return context_history  

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

            # Run inference on the image.
            context_history = None
            if "blip2" not in self.model_name:
                context_history = self.infer_using_blip(img_tensor)
            elif "blip2" in self.model_name:
                context_history = self.infer_using_blip2(img_tensor)
            print(f"Context history: ")
            pprint(context_history)

            # Extract the caption from the output and
            # add to the bottom of the image.
            add_context_history(
                img_path=img_path, 
                context_history=context_history,
                out_dir=out_dir
            )
        else:
            print("Image not found.")

        return

    def get_batch_of_formatted_qa_pairs(self,
                                        batch_qa_pairs):
        """
        This method returns a batch of question-asnwer
        string following a specific format.
        """ 
        batch_formatted_qa_pairs = list(map(
            lambda qa_pair: self.get_formatted_prompt(qa_pair),
            batch_qa_pairs
        ))
        return batch_formatted_qa_pairs

    def get_formatted_context(self,
                              context_history):
        """
        This method returns a formatted context 
        based on the corresponding prompts from the context
        history.
        """
        formatted_context = " ".join([
            context
            for context in context_history
        ])
        return formatted_context

    def get_batch_of_formatted_prompts(self,
                                       batch_cur_ques,
                                       batch_context_histories):
        """
        This method returns a batch of formatted prompts.
        """
        # Get the formatted batch of contexts from the
        # batch of context histories.
        batch_context_histories = list(map(
            lambda context_history: self.get_batch_of_formatted_qa_pairs(context_history), 
            batch_context_histories
        ))
        batch_context_histories = list(map(
            lambda context_history: self.get_formatted_context(context_history),
            list(zip(*batch_context_histories))
        ))

        # Get the formatted batch of current questions and
        # empty answers.
        batch_cur_ques = self.get_batch_of_formatted_qa_pairs(zip(
            batch_cur_ques, 
            self.batch_size * [""]
        ))

        # Get the formatted batch of prompts.
        batch_cur_prompts = list(map(
            lambda x: x[0] + " " + x[1],
            list(zip(batch_context_histories, batch_cur_ques))
        ))

        return batch_cur_prompts

    def infer_batch_using_blip(self,
                               batch_imgs,
                               questions):
        """
        This method runs inference on a batch using the BLIP model.
        """
        batch_context_histories = []
        # Generate prompts and asks questions to
        # the model.
        for i, cur_ques in enumerate(questions):
            cur_ques = cur_ques.strip().lower()
            batch_cur_ques = self.batch_size * [cur_ques]
            output = self.model.predict_answers(
                {"image": batch_imgs, "text_input": batch_cur_ques},
                inference_method="generate"
            )
            batch_context_histories.append(
                tuple(zip(batch_cur_ques, output))
            )
                
        return batch_context_histories

    def infer_batch_using_blip2(self,
                                batch_imgs,
                                questions):
        """
        This method runs inference on a batch using the BLIP-2 model.
        """
        batch_context_histories = []
        # Generate prompts and asks questions to
        # the model.
        for i, cur_ques in enumerate(questions):
            cur_ques = cur_ques.strip().lower()
            batch_cur_ques = self.batch_size * [cur_ques]

            batch_cur_prompts = None
            if i == 0:
                batch_cur_prompts = self.get_batch_of_formatted_qa_pairs(zip(
                    batch_cur_ques, 
                    self.batch_size * [""]
                ))
            else:
                batch_cur_prompts = self.get_batch_of_formatted_prompts(
                    batch_cur_ques=batch_cur_ques,
                    batch_context_histories=batch_context_histories
                )

            # output = self.model.predict_answers(
            #     {"image": batch_imgs, "text_input": batch_cur_ques},
            #     inference_method="generate"
            # )
            output = self.batch_size * ["test_ans"]

            batch_context_histories.append(
                tuple(zip(batch_cur_ques, output))
            )
                
        return batch_context_histories

    def inference_on_batch_of_images(self, 
                                     dataset_path,
                                     question_path,
                                     out_dir):
        """
        This method performs inference on a batch 
        of images.
        TODO: Check if the batch inference method
        works for the BLIP-2 model.
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

        # Load the question set.
        ques_set = load_question_set(question_path)
        num_ques = ques_set["num_questions"]
        questions = ques_set["questions"]

        # Iterate over batches of data.
        for batch_num, (batch_imgs, batch_lbls) in enumerate(tqdm(data_loader, unit="batch")):
            # Generate caption for the batch of images.
            batch_imgs = batch_imgs.to(self.device)

            # Ask questions for the batch of images and 
            # generate the batch of context histories.
            batch_context_histories = None
            if "blip2" not in self.model_name:
                # print("Using BLIP model.")
                batch_context_histories = self.infer_batch_using_blip(
                    batch_imgs=batch_imgs,
                    questions=questions
                )
            elif "blip2" in self.model_name:
                # print("Using BLIP-2 model.")
                batch_context_histories = self.infer_batch_using_blip2(
                    batch_imgs=batch_imgs,
                    questions=questions
                )
            batch_context_histories = list(zip(*batch_context_histories))
            print("batch_context_histories: ")
            pprint(batch_context_histories)
            print("-" * 75)

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
                add_context_history(
                    img_path=img_path, 
                    context_history=batch_context_histories[lbl_num],
                    out_dir=out_par_dir
                )

        return