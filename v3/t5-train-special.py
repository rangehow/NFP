import json
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartTokenizerFast,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    BartTokenizer,
    T5Tokenizer,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from torch.utils.data import Dataset, DataLoader
import datasets

# from parameter import parameter_cnt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pickle
import torch
import argparse
from .dataset import SpecialDataset, SpecialDataCollator
from .special_trainer import KLTrainer
import faulthandler
import ast
from ..config import *

# 启用faulthandler
faulthandler.enable()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
    )
    parser.add_argument("--dataset")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--ce", default=True, type=ast.literal_eval)
    parser.add_argument("--div_mode", default=False, type=ast.literal_eval)
    parser.add_argument("--zero_prob", default=0.1, type=ast.literal_eval)
    parser.add_argument("--clm", default=False, type=ast.literal_eval)
    parser.add_argument("--learning_rate", default=5e-5, type=ast.literal_eval)
    parser.add_argument("--batch_size", default=4)
    return parser.parse_args()


args = parse_args()

print(args)


tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])

collator = SpecialDataCollator(tokenizer)
# 检查数据的调试代码----------------------------------
# from tqdm import tqdm
# with open(f"{args.data_dir}/real_total_dict_train_supervised", "rb") as f:
#     train_dict_supervised = pickle.load(f)

# if args.clm:
#     with open(f"{args.data_dir}/real_total_dict_train_clm", "rb") as f:
#         train_dict_clm = pickle.load(f)
# else:
#     train_dict_clm=None
# train_dataset = SpecialDataset(train_dict_supervised, tokenizer=tokenizer,clm=train_dict_clm, zero_prob=0.1,hybrid=False,quick_mode=True,div_mode=args.div_mode)

# dataloader=tqdm(DataLoader(dataset=train_dataset,batch_size=2,collate_fn=collator,pin_memory=True,num_workers=0))
# for d in dataloader:
#     print(d)
#     # print(d['debug_target'])
#     # exit()
#     # continue
#     exit()
# ------------------------------------------------------


model = T5ForConditionalGeneration.from_pretrained(
    model_dir[args.model], device_map="auto"
)


with open(f"{output_dir[args.dataset]}_supervised", "rb") as f:
    train_dict_supervised = pickle.load(f)

if args.clm:
    with open(f"{output_dir[args.dataset]}_clm", "rb") as f:
        train_dict_clm = pickle.load(f)
else:
    train_dict_clm = None


output_path = args.output_path

train_dataset = SpecialDataset(
    train_dict_supervised,
    tokenizer=tokenizer,
    clm=train_dict_clm,
    zero_prob=args.zero_prob,
    hybrid=False,
    quick_mode=True,
    div_mode=args.div_mode,
)

print("load dataset done")
print(output_path)


trainer = KLTrainer(
    ce=args.ce,
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # optimizers=('adamw_apex_fused',None),
    args=TrainingArguments(
        # optim='adamw_apex_fused',
        overwrite_output_dir=True,
        output_dir=output_path,
        logging_steps=5,
        remove_unused_columns=False,
        gradient_accumulation_steps=32,
        # ------------------------------
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        data_seed=42,
        # eval_delay=100,
        # eval_steps=0.2,
        # -------------------------------
        save_strategy="no",
        # save_steps=0.2,
        # save_total_limit=4,
        # load_best_model_at_end=True,
        # --------------------------------
        dataloader_num_workers=32,
        num_train_epochs=4,
        # auto_find_batch_size=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        bf16=True,
        prediction_loss_only=True,
    ),
    data_collator=collator,
)


trainer.train()
trainer.save_model(output_path)
