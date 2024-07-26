import json
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartTokenizerFast,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from torch.utils.data import Dataset, DataLoader
import datasets
from dataset import MyDataset
from parameter import parameter_cnt
from peft import LoraConfig, get_peft_model
import torch
from config import *


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
    )
    parser.add_argument("--dataset")
    parser.add_argument("--output_path")
    parser.add_argument("--mono")
    return parser.parse_args()


args = parse_args()

model = AutoModelForSeq2SeqLM.from_pretrained(model_dir[args.model], device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
with open(data_dir[args.dataset]) as f:
    train_data = json.load(f)

with open(data_dir[args.dataset]) as f:
    mono_data = json.load(f)


with open(mono_file) as f:
    cnt = 0
    for line in tqdm(f, desc="assist procedure"):
        cnt += 1
        if cnt > 500000:
            break
        line = line.strip()
        assist = tokenizer(line, add_special_tokens=False, return_attention_mask=False)
        # assist = f.read().splitlines()

        assist = [0] + assist["input_ids"] + [tokenizer.eos_token_id]
        d = assist
        length = len(d)
        # print(assist)
        for i in range(length - 1):
            if i > 2:
                clm_key = tuple(d[: i + 1])

                clm_dict[clm_key].update([d[i + 1]])


train_dataset = MyDataset(train_data, tokenizer)

# dataloader=DataLoader(dataset=train_dataset,collate_fn=collator,batch_size=2)
# from tqdm import tqdm
# for d in tqdm(dataloader):
#     print(d['labels'])
#     break
# parameter_cnt(model)


trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # optimizers=('adamw_apex_fused',None),
    args=Seq2SeqTrainingArguments(
        # optim='adamw_apex_fused',
        overwrite_output_dir=True,
        output_dir=args.output_path,
        logging_steps=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        # ------------------------------
        data_seed=42,
        # eval_delay=100,
        # -------------------------------
        save_strategy="no",
        # --------------------------------
        dataloader_num_workers=32,
        num_train_epochs=4,
        # auto_find_batch_size=True,
        per_device_train_batch_size=8,
        bf16=True,
        prediction_loss_only=True,
        # save_safetensors =False,
        # torch_compile=True,
        # torch_compile_backend='inductor',
        # torch_compile_mode='max-autotune',
    ),
    data_collator=collator,
)

# dataloader=trainer.get_train_dataloader()
# from tqdm import tqdm
# for d in tqdm(dataloader):
#     print(d['labels'])
#     exit()
trainer.train(resume_from_checkpoint=False)
trainer.save_model(args.output_path)
