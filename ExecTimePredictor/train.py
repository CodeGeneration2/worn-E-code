# coding = UTF-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from Dataset import MyDatasetFunction
from transformers import DataCollatorForLanguageModeling

model_path = 'codet5-base'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ======================================================== Import Training Data =============================================#
training_data = MyDatasetFunction(
    dataset_path="../../runtime_training_set/train",
    max_tokens=512,
    model_path=model_path,
    train_or_predict="train"
)

# ------------------------------------ Tokenizer ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments
# ############################################################## Training ##################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./trained_model_{model_path}_prompt_runtime_predictor',
    save_strategy="epoch",
    evaluation_strategy="no",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    eval_accumulation_steps=1,

    warmup_steps=500,
    weight_decay=0.01,
    dataloader_num_workers=0,

    logging_dir='./logs',
    logging_strategy="epoch",
    logging_first_step=True,
    save_total_limit=11,
    overwrite_output_dir=True,
    dataloader_drop_last=True,
    dataloader_pin_memory=False,

    prediction_loss_only=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
print('\033[0:34m=========================================== Training Completed ===================================\033[m')
print('\033[0:34m=========================================== Training Completed ===================================\033[m')
print('\033[0:34m=========================================== Training Completed ===================================\033[m')

# trainer.evaluate()
# print('\033[0:34m=========================================== Prediction Completed ===================================\033[m')
# print('\033[0:34m=========================================== Prediction Completed ===================================\033[m')
# print('\033[0:34m=========================================== Prediction Completed ===================================\033[m')

# trainer.save_model(f'model_parameters_{model_path}_prompt')
