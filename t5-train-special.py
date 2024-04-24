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
from parameter import parameter_cnt
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
import pickle
import torch

from dataset import SpecialDataset, SpecialDataCollator
from special_trainer import KLTrainer

tokenizer = AutoTokenizer.from_pretrained(
    "/data/ruanjh/best_training_method/t5v1_1-large"
)
collator = SpecialDataCollator(tokenizer)
# 检查数据的调试代码----------------------------------
# with open('/data/ruanjh/best_training_method/t5/real_total_dict_eval_supervised','rb') as f:
#         total_dict_eval=pickle.load(f)
# eval_dataset=SpecialDataset(total_dict_eval,tokenizer,zero_prob=0)
# dataloader=DataLoader(dataset=eval_dataset,batch_size=2,collate_fn=collator,)
# for d in dataloader:
#     # print(d)
#     # print(d['target'].shape)
#     import pdb
#     pdb.set_trace()
#     # exit()
#     print(1)
#     continue
# ------------------------------------------------------
# config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )
model = T5ForConditionalGeneration.from_pretrained(
    "/data/ruanjh/best_training_method/t5v1_1-large", 
#     quantization_config=config
)
model.save_pretrained('?')
exit()
# model = prepare_model_for_kbit_training(model)
# collator= DataCollatorForSeq2Seq(tokenizer,model=model,padding=True)

# collator([dataset[0],dataset[1]])
with open("/data/ruanjh/best_training_method/t5/real_total_dict__supervised", "rb") as f:
    total_dict = pickle.load(f)
with open("/data/ruanjh/best_training_method/t5/real_total_dict_eval_supervised", "rb") as f:
    total_dict_eval = pickle.load(f)


train_dataset = SpecialDataset(total_dict, tokenizer, zero_prob=0.1)
eval_dataset = SpecialDataset(total_dict_eval, tokenizer, zero_prob=0.1)
print(len(train_dataset))


# print(model)
# dataloader=DataLoader(dataset=eval_dataset,batch_size=2,collate_fn=collator)
# for d in dataloader:
#     print(d)
#     exit()

# lora_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     task_type="SEQ_2_SEQ_LM",
#     bias="none",
#     use_rslora=True,
# )
# model.add_adapter(lora_config)
# model = get_peft_model(model, lora_config)
# parameter_cnt(model)

trainer = KLTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # optimizers=('adamw_apex_fused',None),
    args=TrainingArguments(
        optim='adamw_apex_fused',
        overwrite_output_dir=True,
        output_dir="/data/ruanjh/best_training_method/T5_V05",
        logging_steps=5,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        # ------------------------------
        evaluation_strategy="steps",
        # eval_delay=100,
        eval_steps=100,
        # -------------------------------
        save_strategy="epoch",
        # save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        # --------------------------------
        dataloader_num_workers=0,
        num_train_epochs=4,
        # auto_find_batch_size=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        bf16=True,
        prediction_loss_only=True,
    ),
    data_collator=collator,
)

trainer.train()
