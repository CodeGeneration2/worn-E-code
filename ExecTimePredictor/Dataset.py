# coding=utf-8
"""
Dataset to be used for APPS Training
Note: Successful GPT
"""
from statistics import mean

import torch

import os

from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
import random
import json


# ################################################################# MyDatasetFunction #####################################
class MyDatasetFunction(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_tokens=512, model_path='codet5-base', train_or_predict=""):

        self.dataset_path = dataset_path
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.train_or_predict = train_or_predict

        # ------------------------------------ Tokenizer ------------------------------------#
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.total_data_list = []  # Should be set in init_function()
        self.total_data_dict = {}  # Should be set in init_function()

        # ========================= Init Function (Load data from local) ===============#
        self.init_function()

    # =========================================== Init Function (Load data from local) =========================================#
    def init_function(self):
        """
        Load data from local
        Returns:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_root_path is set to folderName/data
        """

        total_data_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")
        for question in tqdm(dataset_list):
            # ============================================================= Feature ====================================#
            code_list = os.listdir(f"{self.dataset_path}/{question}")
            for code in code_list:
                with open(f"{self.dataset_path}/{question}/{code}", 'r', encoding='UTF-8') as f:
                    content = f.read()

                input_feature = f"{content}\nEvaluate the code running time:"

                run_time = str(int(code.split("KB, standard_time ")[-1].split(" ms,")[0]))

                # ---------------------------------------- Min Unit ----------------------------------------#
                data_tuple = (input_feature, run_time, f"{self.dataset_path}/{question}/{code}")
                # -------------------------- Append to total_data_list -----------#
                total_data_list.append(data_tuple)

        # ----------------------- Randomize List - Comparative experiment -------------#
        random.shuffle(total_data_list)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== Loaded {len(total_data_list)} training data items ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iteration function ==================================#
    def __getitem__(self, index):
        input_feature, run_time, code_path = self.total_data_list[index]

        # ---------------------------- Will never delete. Defer bug fixing. --------------#
        input_feature = input_feature[:150000]
        run_time = run_time[:150000]

        # ------------------------------------- Encode ---------------------------------------------#
        feature_encoded_dict = self.tokenizer(input_feature, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")

        # ------------------------------------- Encode ---------------------------------------------#
        label_encoded_dict = self.tokenizer(run_time, padding='max_length', max_length=6, return_tensors="pt")

        if self.train_or_predict == "train":
            # ------------------------------------- Encode ---------------------------------------------#
            feature_encoded_dict["input_ids"] = feature_encoded_dict['input_ids'].squeeze()
            feature_encoded_dict["attention_mask"] = feature_encoded_dict['attention_mask'].squeeze()
            feature_encoded_dict["labels"] = label_encoded_dict['input_ids'].squeeze()

        elif self.train_or_predict == "predict":
            # ------------------------------------- Encode ---------------------------------------------#
            feature_encoded_dict["input_feature"] = input_feature
            feature_encoded_dict["run_time"] = run_time
            feature_encoded_dict["code_path"] = code_path

        else:
            print("Error in self.train_or_predict!!!!!!!")

        if "incoder" in self.model_path:
            feature_encoded_dict.pop("token_type_ids", None)

        return feature_encoded_dict


# ==================================================================================#
# ==================================================================================#
# ==================================================================================#
if __name__ == '__main__':
    test_data = MyDatasetFunction(
        dataset_path=r"F:\0Work3: Expert Group-High Efficiency Code Generation-Paper-Fourth Rejection\5 - Fifth Submission-Split-Pure Expert Group\dataset\7-Run Time Training Set-5 and 6-AST Merged Version\train",
        max_tokens=512,
        model_path=r"F:\PythonPureData\LargeLanguageModelParams\codet5-base",
        train_or_predict="train"
    )
    NPI_list = []
    for data in tqdm(test_data):
        if len(data["labels"]) > 6:
            print(len(data["labels"]))

