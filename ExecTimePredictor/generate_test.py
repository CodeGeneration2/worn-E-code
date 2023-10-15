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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_path = './trained_model_codet5-base_prompt_runtime_predictor/checkpoint-189160'
# model_path = './CodeBERT-base'
# model_path = r'F:\ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted\generated-code-and-trained-model\trained_model_parameters_CodeBERT_NPI_score\checkpoint-35976'
# model_path = r'F:\ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted\generated-code-and-trained-model\trained_model_parameters_CodeBERT_NPI_score\checkpoint-35976'

generated_code_dir = "../../runtime_training_set/test"

# ------------------------------------ Tokenization Dictionary ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
model.to(device)

def generate_runtime(code_text):
    test_tensor = tokenizer(code_text, truncation=True, max_length=512, return_tensors="pt")
    # print(test_tensor)
    with torch.no_grad():
        generated_token_list = model.generate(test_tensor['input_ids'].to(device))

    runtime = tokenizer.decode(generated_token_list[0], skip_special_tokens=True)

    return runtime

# #######################################################################################################
def total_evaluation_function(evaluation_set_path):
    error_list = []

    problem_list = os.listdir(evaluation_set_path)
    for problem_index in tqdm(problem_list):
        # if int(problem_index) < 8000:
        #     continue

        code_list = os.listdir(f"{evaluation_set_path}/{problem_index}")
        for code in code_list:
            # ---------------------------- Save record file -----------------------------------------------------------#
            with open(f"{evaluation_set_path}/{problem_index}/{code}", 'r', encoding='UTF-8') as f:
                code_text = f.read()

            # if "1 new NPI new 1" in code:
            #     continue

            runtime = generate_runtime(code_text)
            label_time = code.split("KB,standard_time ")[-1].split(" ms,NPI")[0]

            try:
                error_value = int(runtime) - int(label_time)
                error_list.append(error_value)
            except:
                runtime = "Cannot predict"
                error_value = "Cannot predict"

            new_name = code.replace(".txt", f",predicted {runtime} time,error {error_value} value.txt")

            os.rename(f"{evaluation_set_path}/{problem_index}/{code}", f"{evaluation_set_path}/{problem_index}/{new_name}")

    return mean(error_list)


# ############################################################ Entry Point ####################################################
if __name__ == '__main__':
    generated_code_list = os.listdir(generated_code_dir)
    for generated_code_set in generated_code_list:
        average_error = total_evaluation_function(evaluation_set_path=f"{generated_code_dir}/{generated_code_set}")
        print(f"Average error: {average_error}. Model path: {generated_code_dir}/{generated_code_set}")


