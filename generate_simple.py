from transformers import BartForConditionalGeneration,BartTokenizerFast
import json
from transformers import BartConfig,BartForConditionalGeneration,Seq2SeqTrainer,DataCollatorForSeq2Seq,BartTokenizerFast,TrainingArguments,Seq2SeqTrainingArguments,BartTokenizer,AutoModelForSeq2SeqLM,AutoTokenizer,DefaultDataCollator
from torch.utils.data import Dataset,DataLoader

import torch

from tqdm import tqdm
import json

    
class MyDataset(Dataset):
    def __init__(self,data,tokenizer):
        if isinstance(data[0],dict):
            src=[d['en'] for d in data]
            
        else:
            src=[d[:-1] for d in data]
        
        input_id=tokenizer(src,return_tensors='pt',padding=True,add_special_tokens=False)
        # print(input_id)
        self.data=input_id
        
        

        
    def __getitem__(self, index):

        return {'input_ids':self.data['input_ids'][index],
                'attention_mask':self.data['attention_mask'][index]}

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data['input_ids'])

import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag') 
    return parser.parse_args()


args=parse_args()
    

if __name__=='__main__':
    with torch.inference_mode():
        tag=args.tag 
        model_dir=f'/data/ruanjh/{tag}'
        tokenizer=AutoTokenizer.from_pretrained(model_dir)
        model=AutoModelForSeq2SeqLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,).to_bettertransformer()
        model.cuda()
        for task in ['wmt22','iwslt17']:
            if task=='iwslt17':
                with open('/data/ruanjh/best_training_method/iwslt17/test.json') as f:
                    test_data=json.load(f) 
            else: 
                with open('/data/ruanjh/best_training_method/wmt22/generaltest2022.en-de.src.en') as f:
                    test_data=f.readlines()
            
            data_all = test_data
            aaaa=tag.replace('/','_')
            out_path=f'/data/ruanjh/best_training_method/{task}/{aaaa}'
            
            print('model_dir:',model_dir,'out_path:',out_path)
            
            dataset=MyDataset(data_all,tokenizer)
            collator= DefaultDataCollator()
            dataloader=DataLoader(dataset,16,collate_fn=collator,pin_memory=True,num_workers=8)
            result=[]
            for input in tqdm(dataloader,desc=f'{task}'):

                
                output = model.generate(
                            input_ids=input['input_ids'].to('cuda'),
                            attention_mask=input['attention_mask'].to('cuda'),
                            num_beams=5,
                            do_sample=False,
                            temperature=1.0,
                            max_new_tokens=256,
                        )
                pred = tokenizer.batch_decode(output,skip_special_tokens=True)
                result+=pred
            print('output',out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                for r in result:
                    f.write(r.replace('\n','\\n')+'\n')