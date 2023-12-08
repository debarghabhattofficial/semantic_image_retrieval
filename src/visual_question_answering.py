import os
import argparse
from tqdm import tqdm
from pprint import pprint
import re

import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool
from torch.multiprocessing import spawn, Process, set_start_method
from lavis.models import load_model_and_preprocess

from concurrent.futures import ThreadPoolExecutor

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
        self.num_threads = torch.multiprocessing.cpu_count()

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

    def is_alphanumeric_answer(self, answer):
        # Define a regular expression pattern for words 
        # containing alphabets and digits.
        pattern = re.compile(r"^[a-zA-Z0-9\s]+$")

        # Use the pattern to match the generated answer.
        match = pattern.match(answer)

        # If there's a match, the string is made only 
        # of words with alphabets and digits.
        return bool(match)

    def infer_using_blip2(self,
                          img_tensor):
        """
        This method runs inference using the BLIP-2 model.
        TODO: Change the method to follow similar procedure as 
        one used for running batch inference using BLIP-2.
        """
        # We maintain two list for storing the context histories-
        # 'cur_context_history' keeps track of the current active 
        # context and 'context_history''keeps track of the entire 
        # context history.
        context_history = []
        cur_context_history = []
        # Generate prompts and asks questions to the model.
        output = None
        while True:
            cur_ques = input("Enter question: ")
            cur_ques = str(cur_ques).strip().lower()
            cur_prompt = None
            if (output is None) or (not self.is_alphanumeric_answer(output[:-1])):
                # If the last output generated by the model is
                # not alphanumeric, we reset the context to start
                # from the current question.
                cur_prompt = self.get_formatted_prompt((cur_ques, ""))
                context_history.extend(cur_context_history)
                cur_context_history = []
            else:
                # If the last output generated by the model is
                # alphanumeric, we append the current question
                # to the existing context and continue.
                cur_prompt = " ".join(
                    list(map(
                        lambda context: self.get_formatted_prompt(context), 
                        cur_context_history
                    ))
                ) + " " + self.get_formatted_prompt((cur_ques, ""))
            cur_ans = ""
            output = self.model.generate(
                {"image": img_tensor, "prompt": cur_prompt},
            )[0]
            if (output == "") or (output[-1] != "."):
                # Append a full stop at the end of the output.
                # This is essential to signify the end of a 
                # question-answer pair in next iteration's prompt.
                output += "."
            cur_ans = output
            cur_context_history.append((cur_ques, cur_ans))
            cnt_msg = "Press 'N' to abort, " + \
                "or any other key to ask another question: "
            cnt = input(cnt_msg)
            cnt = str(cnt)
            if (cnt == "N") or (cnt == "n"):
                break
                
        context_history.extend(cur_context_history)
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
                # Use the BLIP model to ask isolated questions
                # (standard VQA). 
                context_history = self.infer_using_blip(img_tensor)
            elif "blip2" in self.model_name:
                # Use the BLIP-2 model to ask questions with 
                # previous question-answer pairs (instructed VQA).
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

    def get_batch_of_formatted_qa_pairs(self,
                                        batch_qa_pairs):
        """
        This method returns a batch of question-answer
        string following a specific format.
        """
        batch_formatted_qa_pairs = list(map(
            lambda qa_pair: self.get_formatted_prompt(qa_pair),
            batch_qa_pairs
        ))
        return batch_formatted_qa_pairs

    def get_formatted_context(self, x):
        """
        This method returns a formatted context 
        based on the corresponding prompts from the context
        history.
        """
        context_history, window_start_idx = x
        formatted_context = " ".join([
            context_history[i]
            for i in range(window_start_idx, len(context_history))
        ])
        return formatted_context

    def get_batch_of_formatted_prompts(self,
                                       batch_cur_ques,
                                       batch_context_histories,
                                       batch_window_start_indices):
        """
        This method returns a batch of formatted prompts.
        """
        # Get the formatted batch of contexts from the
        # batch of context histories.
        batch_context_histories = list(map(
            lambda context_history: self.get_batch_of_formatted_qa_pairs(context_history), 
            list(zip(*batch_context_histories))
        ))
        batch_context_histories = list(map(
            lambda x: self.get_formatted_context(x),
            list(zip(batch_context_histories, batch_window_start_indices))
        ))

        # Get the formatted batch of current questions and
        # empty answers.
        batch_cur_ques = self.get_batch_of_formatted_qa_pairs(list(zip(
            batch_cur_ques, 
            self.batch_size * [""]
        )))

        # Get the formatted batch of prompts.
        batch_cur_prompts = list(map(
            lambda x: (x[0] + " " + x[1]).strip(),
            list(zip(batch_context_histories, batch_cur_ques))
        ))

        return batch_cur_prompts

    def ask_blip2(self, input_data):
        """
        This method asks a question to the BLIP-2 model.
        """
        img_tensor, prompt = input_data
        img_tensor = img_tensor.unsqueeze(0)
        # Get the model's response for the input
        # image-prompt input.
        output = self.model.generate(
            {"image": img_tensor, "prompt": prompt},
        )[0]
        if (output == "") or (output[-1] != "."):
            # Append a full stop at the end of the output.
            # This is essential to signify the end of a 
            # question-answer pair in next iteration's prompt.
            output += "."
        return output

    def update_window_start_indices(self,
                                    output):
        """
        This methods updates the window start indices
        based on the output.
        """
        # Flag to indicate if the window start indices 
        # need to be updated.
        update_window = False
        if (output is None) or (not self.is_alphanumeric_answer(output[:-1])):
            update_window = True
        return update_window

    def infer_batch_using_blip2(self,
                                batch_imgs,
                                questions):
        """
        This method runs inference on a batch using the BLIP-2 model.
        """
        context_histories = [[] * self.batch_size]
        batch_context_histories = []
        # Keeps track of start indices of different windows.
        batch_window_start_indices = [0] * self.batch_size
        # Generate prompts and asks questions to
        # the model.
        for ts, cur_ques in enumerate(questions):
            cur_ques = cur_ques.strip().lower()
            batch_cur_ques = self.batch_size * [cur_ques]

            batch_cur_prompts = None
            if ts == 0:
                # The initial prompt is only the current question.
                batch_cur_prompts = self.get_batch_of_formatted_qa_pairs(list(zip(
                    batch_cur_ques, 
                    self.batch_size * [""]
                )))
            else:
                # Subsequent prompts are the concatenation of the
                # previous question and answer pairs and the current
                # question.
                batch_cur_prompts = self.get_batch_of_formatted_prompts(
                    batch_cur_ques=batch_cur_ques,
                    batch_context_histories=batch_context_histories,
                    batch_window_start_indices=batch_window_start_indices
                )

            # Perform inference in parallel using ThreadPoolExecutor
            batch_outputs = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                batch_outputs = list(executor.map(
                    self.ask_blip2, 
                    list(zip(batch_imgs, batch_cur_prompts))
                ))

            # Find if the window start indices need to be updated
            # based on the output.
            update_windows_on = list(map(
                lambda output: self.update_window_start_indices(output),
                batch_outputs
            ))
            # Update the window start indices.
            for k in range(self.batch_size):
                if update_windows_on[k]:
                    batch_window_start_indices[k] = ts + 1

            # Update the batch of context histories.
            batch_context_histories.append(
                tuple(zip(batch_cur_ques, batch_outputs))
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
                # Use the BLIP model to ask isolated questions
                # (standard VQA). 
                batch_context_histories = self.infer_batch_using_blip(
                    batch_imgs=batch_imgs,
                    questions=questions
                )
            elif "blip2" in self.model_name:
                # Use the BLIP-2 model to ask questions with 
                # previous question-answer pairs (instructed VQA).
                batch_context_histories = self.infer_batch_using_blip2(
                    batch_imgs=batch_imgs,
                    questions=questions
                )
            batch_context_histories = list(zip(*batch_context_histories))

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