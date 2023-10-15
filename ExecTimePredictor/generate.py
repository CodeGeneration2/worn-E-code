# -*- coding: utf-8 -*-
import os
from statistics import mean

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# #####################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_path = './trained_model_codet5-base_prompt_runtime_predictor/checkpoint-189160'
# model_path = './CodeBERT-base'
# model_path = r'F:\ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted\generated-code-and-trained-model\trained_model_parameters_CodeBERT_NPI_score\checkpoint-35976'
# model_path = r'F:\ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted\generated-code-and-trained-model\trained_model_parameters_CodeBERT_NPI_score\checkpoint-35976'

generated_code_path = "../../trained_model_parameters_PolyCoder-0.4B_GPT_prompt_generate_code"

code_path_dataset_size_1 = 1

# ------------------------------------ Tokenization Dictionary ------------------------------------#
tokenization_dict = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
model.to(device)


def generate_runtime(code_text):
    test_tensor = tokenization_dict(code_text, truncation=True, max_length=512, return_tensors="pt")
    # =================================================== simply generate a single sequence ===========================
    with torch.no_grad():
        generated_token_list = model.generate(test_tensor['input_ids'].to(device))

    runtime = tokenization_dict.decode(generated_token_list[0], skip_special_tokens=True)

    return runtime


# #######################################################################################################
def total_evaluation_function(evaluation_set_path):
    error_list = []

    problem_list = os.listdir(evaluation_set_path)
    for problem_index in tqdm(problem_list):

        code_set_name_list = os.listdir(f"{evaluation_set_path}/{problem_index}")
        for code_set in code_set_name_list:

            code_list = os.listdir(f"{evaluation_set_path}/{problem_index}/{code_set}")
            for code in code_list:

                # ---------------------------- Save record file -----------------------------------------------------------#
                with open(f"{evaluation_set_path}/{problem_index}/{code_set}/{code}", 'r', encoding='UTF-8') as f:
                    code_text = f.read()

                # if "1 new NPI new 1" in code:
                #     continue

                runtime = generate_runtime(code_text)
                try:
                    error_list.append(int(runtime))
                except:
                    runtime = "Cannot predict"

                new_name = code.replace(".txt", f", predicted time {runtime} ms.txt")

                os.rename(f"{evaluation_set_path}/{problem_index}/{code_set}/{code}", f"{evaluation_set_path}/{problem_index}/{code_set}/{new_name}")

    return mean(error_list)


# ############################################################ Entry Point ####################################################
if __name__ == '__main__':
    if code_path_dataset_size_1 == 1:
        average_value = total_evaluation_function(evaluation_set_path=f"{generated_code_path}")
        print(f"Average value: {average_value}. Model path: {generated_code_path}")
    else:
        generated_code_list = os.listdir(generated_code_path)
        for generated_code_set in generated_code_list:

            average_value = total_evaluation_function(evaluation_set_path=f"{generated_code_path}/{generated_code_set}")

            print(f"Average value: {average_value}. Model path: {generated_code_path}/{generated_code_set}")


