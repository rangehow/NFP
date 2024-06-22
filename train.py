import json
from transformers import BartConfig,BartForConditionalGeneration,Seq2SeqTrainer,DataCollatorForSeq2Seq,BartTokenizerFast,TrainingArguments,Seq2SeqTrainingArguments,BartTokenizer,AutoTokenizer,AutoModelForSeq2SeqLM
from torch.utils.data import Dataset,DataLoader
import datasets
from dataset import MyDataset
from parameter import parameter_cnt
from peft import LoraConfig,get_peft_model
import torch
tokenizer=AutoTokenizer.from_pretrained('/data/ruanjh/best_training_method/t5v1_1-large')
with open('/data/ruanjh/best_training_method/iwslt17/train.json') as f:
    train_data=json.load(f)
with open('/data/ruanjh/best_training_method/iwslt17/validation.json') as f:
    eval_data=json.load(f)


train_dataset=MyDataset(train_data,tokenizer)
eval_dataset=MyDataset(eval_data,tokenizer)

# from torch.utils.data import RandomSampler
# def get_train_sampler(train_dataset) :
#     return RandomSampler(train_dataset)

# from transformers.trainer_utils import set_seed

# set_seed(42)
# a=get_train_sampler(train_dataset)
# cnt=0
# for i in a:
#     print(i)
#     cnt+=1
#     if cnt>2:
#         break
# import pdb
# pdb.set_trace()

model=AutoModelForSeq2SeqLM.from_pretrained('/data/ruanjh/best_training_method/t5v1_1-large')
collator= DataCollatorForSeq2Seq(tokenizer,model=model,padding=True)

# dataloader=DataLoader(dataset=train_dataset,collate_fn=collator,batch_size=2)
# from tqdm import tqdm
# for d in tqdm(dataloader):
#     print(d['labels'])
#     break
# parameter_cnt(model)



trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # optimizers=('adamw_apex_fused',None),
        args=Seq2SeqTrainingArguments(
            # optim='adamw_apex_fused',
            overwrite_output_dir =True,
            output_dir="/data/ruanjh/best_training_method/t5-raw-8card",
            logging_steps=1,
            remove_unused_columns =False,
            gradient_accumulation_steps=16,
            #------------------------------
            evaluation_strategy='steps',
            data_seed =42,
            # eval_delay=100,
            eval_steps =100,
            #-------------------------------
            save_strategy ='steps',
            save_steps = 100,
            save_total_limit =3,
            load_best_model_at_end=True,
            #--------------------------------
            dataloader_num_workers =8,
            num_train_epochs=4,
            # auto_find_batch_size=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size =2,
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
trainer.save_model('/data/ruanjh/best_training_method/t5-raw-8card')