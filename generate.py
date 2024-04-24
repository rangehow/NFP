from transformers import BartForConditionalGeneration,BartTokenizerFast
import json
from transformers import BartConfig,BartForConditionalGeneration,Seq2SeqTrainer,DataCollatorForSeq2Seq,BartTokenizerFast,TrainingArguments,Seq2SeqTrainingArguments,BartTokenizer,AutoModelForSeq2SeqLM,AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import datasets
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
# params = list(model.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))

import json

    
class MyDataset(Dataset):
    def __init__(self,data,tokenizer):
        if isinstance(data[0],dict):
            src=[d['en'] for d in data]
            tgt=[d['de'] for d in data]
            input_id=tokenizer(src)
            tgt_id=tokenizer(tgt)
            self.data={'input_ids':input_id.input_ids,'attention_mask':input_id.attention_mask,
                   'labels':tgt_id.input_ids,'decoder_attention_mask':tgt_id.attention_mask} 
        else:
            src=[d[:-1] for d in data]
            input_id=tokenizer(src)
            self.data={'input_ids':input_id.input_ids,'attention_mask':input_id.attention_mask,
                   } 
        
        

        
    def __getitem__(self, index):
        return {'input_ids':self.data['input_ids'][index],'index':[index]
                #    'labels':self.data['labels'][index],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data['input_ids'])


    def __init__(self,data,tokenizer):
        src=[d['en'] for d in data]
        tgt=[d['de'] for d in data]
        input_id=tokenizer(src)
        tgt_id=tokenizer(tgt)

        self.data={'input_ids':input_id.input_ids,'attention_mask':input_id.attention_mask,
                   'labels':tgt_id.input_ids,'decoder_attention_mask':tgt_id.attention_mask} 
    def __getitem__(self, index):
        return {'input_ids':self.data['input_ids'][index],'index':[index]
                #    'labels':self.data['labels'][index],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data['input_ids'])


def load_model_and_tokenizer(device,model_dir='/data/ruanjh/best_training_method/T5_V3_55/checkpoint-5500'):
    model=AutoModelForSeq2SeqLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    print('device',device)
    model.to(device)
    tokenizer=AutoTokenizer.from_pretrained(model_dir)
    return model,tokenizer

def get_pred(rank,out_path,data,dict):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(device)
    # print(tokenizer.special_token)
    # 这里直接用mydataset没关系，因为这是t5类型，decoder那边会自己给个[EOS]开头吧
    dataset=MyDataset(data,tokenizer)
    collator= DataCollatorForSeq2Seq(tokenizer,model=model,padding=True)
    dataloader=DataLoader(dataset,8,collate_fn=collator,pin_memory=True,num_workers=0)
    result=[]
    for input in tqdm(dataloader):
        input.to(device)
        output = model.generate(
                    input_ids=input['input_ids'],
                    attention_mask=input['attention_mask'],
                    num_beams=5,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=512,
                )
        pred = tokenizer.batch_decode(output,skip_special_tokens=True)
        result+=pred
    dict[f'{rank}']=result
    
    # dist.destroy_process_group()
    
def split_list(lst, n):
    avg = len(lst) / float(n)
    return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

if __name__=='__main__':
    with open('/data/ruanjh/best_training_method/iwslt17/test.json') as f:
        test_data=json.load(f)
    # with open('/data/ruanjh/best_training_method/wmt22/generaltest2022.en-de.src.en') as f:
    #     test_data=f.readlines()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    data_all = [data_sample for data_sample in test_data]
    data_subsets = split_list(data_all,world_size)
    out_path='/data/ruanjh/best_training_method/iswlt17/t5_v3_55testparallel'
    processes = []
    manager = mp.Manager()
    dict = manager.dict()
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank,out_path,data_subsets[rank],dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(out_path, "w", encoding="utf-8") as f:
        for rank in range(world_size):
            for r in dict[f'{rank}']:
                f.write(r.replace('\n','\\n')+'\n')