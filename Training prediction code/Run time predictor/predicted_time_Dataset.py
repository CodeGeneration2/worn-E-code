# coding=utf-8
"""
Dataset to be used for APPS Training
"""

import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from dataset_lm.reindent import run as run_reindent
from tqdm import tqdm


import json

# ############################################################# 我的Dataset函数 #####################################
class 训练集Dataset类(torch.utils.data.Dataset):
    def __init__(self, 模型="EleutherAI/gpt-neo-125M", 最大Token数=2046, 数据集根路径="测试集"):

        self.模型 = 模型

        self.Token化词表 = transformers.GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")

        self.总数据列表 = []  # Should be set in 初始化函数()

        # ================================= 初始化函数（将数据从本地导入） ==================#
        self.从当前目录初始化函数(数据集根路径)

    # =========================================== 初始化函数（将数据从本地导入） =========================================#
    def 从当前目录初始化函数(self, 数据集根路径):
        """ 从本地导入数据
        返回：
            self.总数据列表 = 总数据列表
        """

        总数据列表 = []

        训练集列表 = os.listdir(f"{数据集根路径}")
        # ----------------------------------------- 导入数据 ------------------------------------------------------------#
        print('\033[0:34m========================== 导入 训练集 数据中... (具体是 将数据转化为ID) ==================\033[m')
        for 某条 in tqdm(训练集列表):
        # for 某条 in tqdm(range(2000)):
        #for 某条 in tqdm(range(4)):
            if 某条.split(",")[1]=="BLEU分数":
                # ----------------------------------------------- 标签 --------------------------------#
                with open(f"{数据集根路径}/{某条}", 'r', encoding='UTF-8') as f:
                    代码 = f.read()

                # ------------------------------ 加入总数据列表 -----------#
                总数据列表.append(代码)

        print(f'\033[0:35m========================== 已加载 {len(总数据列表)} 条 训练集 数据 ==================\033[m')

        self.总数据列表 = 总数据列表


    def __len__(self):
        return len(self.总数据列表)

    # ========================================= 迭代遍历函数 =========================================#
    def __getitem__(self, 索引):

        # ----------(标题,问题描述主体,Input描述,Output描述,输入输出样例测试,Note描述, 某慢速代码, 标签代码, 数据原路径)--------#
        样本列表 = self.总数据列表[索引]

        return 样本列表


def 缩进代码函数(代码字符串):
    """
    给定的代码字符串，以 Github 的方式 重新缩进它
    Given code string, reindent it in the same way that the Github 数据集 was indented
    """
    # --------------------------------- 可变字符串_io.stringIO操作 ---------------------------------#
    代码字符串 = io.StringIO(代码字符串)
    缩进后代码字符串 = io.StringIO()

    run_reindent(
        代码字符串,
        缩进后代码字符串,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    # ------------------- 获取对象值 ---------------#
    return 缩进后代码字符串.getvalue()


if __name__ == '__main__':

    Token化词表 = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    训练集 = 训练集Dataset类(
        模型="EleutherAI/gpt-neo-125M",
        最大Token数=2048
    )

    e = 训练集[0]
    print(e)
    print("------- input_ids ------------------------------------------------------------------------------------")
    print(Token化词表.decode(e['input_ids']))

    print("------- labels ------------------------------------------------------------------------------------")
    labels = e['labels']
    print(f"原始标签：{labels}")

    for 某id in range(len(labels)):
        if labels[某id] == -100:
            labels[某id] = Token化词表.eos_token_id
    print(f"一步处理标签：{labels}")

    labels_str = Token化词表.decode(labels)
    print(f"标签：{labels_str}")