import time
import torch.multiprocessing as mp
import torch

mp.set_sharing_strategy("file_system")
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import pickle
import pdb
from copy import deepcopy
from scipy.optimize import fsolve
import numpy as np
import warnings
# from dataset import MyDataset, PreprocessDataset
from tqdm import tqdm
warnings.filterwarnings("ignore", "The iteration is not making good progress")
import sys



from torch.utils.data import Dataset


from functools import partial
def wmt19(example,tokenizer):
    src = [d["en"] for d in example['translation']]
    tgt = [d["de"] for d in example['translation']]
    input_id = tokenizer(src,add_special_tokens=False,return_attention_mask=False)
    tgt_id = tokenizer(tgt,add_special_tokens=False,return_attention_mask=False)

    result_matrix =[[0]+ row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
    data = {
        "input_ids": input_id.input_ids, 
        "labels": result_matrix,
    }

    return data


def iwslt17(example,tokenizer):
    src = example['en']
    tgt = example['de']
    input_id = tokenizer(src,add_special_tokens=False,return_attention_mask=False)
    tgt_id = tokenizer(tgt,add_special_tokens=False,return_attention_mask=False)

    result_matrix =[[0]+ row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
    data = {
        "input_ids": input_id.input_ids, 
        "labels": result_matrix,
    }

    return data

def ugly_allocate(example,tokenizer):
    if 'translation' in example:
        return wmt19(example,tokenizer)
    elif 'en' in example:
        return iwslt17(example,tokenizer)

import datasets
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dir',default="/data/ruanjh/best_training_method/t5v1_1-large") 
    parser.add_argument('--data_dir',default='/data/ruanjh/wmt19')
    parser.add_argument('--mono_file',default='/data/ruanjh/news.2023.de.shuffled.deduped')
    parser.add_argument('--clm_mode',default=True)
    parser.add_argument('--mono',default=True)
    parser.add_argument('--cache_dir',default='/data/ruanjh/cache')
    parser.add_argument('--trie',action='store_true')
    parser.add_argument('--ngram',default=5)
    return parser.parse_args()

args=parse_args()

if args.trie:
    from trie import Trie
    total_trie=Trie()
    clm_trie=Trie()

else:
    total_dict = defaultdict(Counter)
    clm_dict = defaultdict(Counter)

def generate(example):
    # print('?',example['labels'][0],rank)

    for j in range(len(example['labels'])):

        length = len(example['labels'][j])
        for i in range(length - 1):
            # value总是预测的下一个词
            value = example['labels'][j][i + 1]
            decoder_input_id = example['labels'][j][: i + 1]
            input_id = example['input_ids'][j]

            if args.trie:
                # 不加-1也能区分开 Input_id就长某个input_id+decoder_input_id的情况，因为decoder开头是1，这个是input_id不可能具有的内容。
                key = input_id+decoder_input_id
                total_trie.update(key,value)
            else:
                key=(tuple(input_id), tuple(decoder_input_id))
                total_dict[key].update([value])
                 
                    
            if args.clm_mode and i>args.ngram-2: # >k就是 k+2 gram，
                if args.trie:
                    clm_trie.update(decoder_input_id,value)
                else:
                    clm_key = tuple(decoder_input_id)
                    clm_dict[clm_key].update([value])

                

real_total_dict = defaultdict(list)
if args.clm_mode:
    real_total_clm_dict= defaultdict(list)          
def reformate(example):
    for j in range(len(example['labels'])):
        d = train_dateset[j]

        decoder_input_id = d["labels"]
        input_id = d["input_ids"]
        # 这里要丢掉key里decoder_input_id最后的EOS，因为只有label需要最后是EOS，下面到length-1也是因为不需要考虑EOS位置的后续词（本来就没有）
        key = (tuple(input_id), tuple(decoder_input_id[:-1]))
        length = len(d["labels"])


        for i in range(length - 1):

            temp_key = (tuple(input_id), tuple(decoder_input_id[: i + 1]))

            if (
                len(real_total_dict[key]) < len(decoder_input_id) - 1 #防止重复
                and temp_key in total_dict
            ):
                temp_value = total_dict[temp_key]

                if args.clm_mode:
                    
                    clm_temp_key = tuple(decoder_input_id[: i + 1])
                    if clm_temp_key not in clm_dict:
                        # 如果clm这个分布是没有的，那就直接用supervised就好了。
                        # 反正是加权平均，相当于把clm这个置为0了
                        real_total_clm_dict[key].append(temp_value)
                    else:
                        real_total_clm_dict[key].append(clm_dict[clm_temp_key])

                real_total_dict[key].append(temp_value)

            else:
                break
if __name__ == "__main__":

    clm_mode = args.clm_mode
    mono=args.mono
    file = [
        ("train", "train"),
    ]
    mono_file=args.mono_file
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,add_special_tokens=False,
    )
    # debug_cnt=0
    for input, output in file:
        
        # with open(f"/data/ruanjh/best_training_method/iwslt17/{input}.json") as f:
        #     train_data = json.load(f)
        
        # train_dateset = PreprocessDataset(train_data, tokenizer)
        if args.data_dir.endswith('.json'):
            train_dateset=datasets.load_dataset('json',data_files=args.data_dir)[f'{input}']

        else:
            train_dateset=datasets.load_from_disk(args.data_dir)[f'{input}'][:3000000]
        train_dateset=train_dateset.map(partial(ugly_allocate,tokenizer=tokenizer),remove_columns=train_dateset.features.keys(),batched=True,num_proc=96,load_from_cache_file=False)
        
        train_dateset.map(partial(generate),batched=True,num_proc=1)
        train_dateset.map(partial(reformate),batched=True,num_proc=50)
        # train_dateset=train_dateset.map(partial(generate,clm_mode=clm_mode,mono=mono),batched=True,cache_file_name=f'{args.data_dir}/{input}_final.cache',num_proc=50,load_from_cache_file=True)
        # dataloader=DataLoader(train_dateset,batch_size=10000,collate_fn=lambda x:x)
        # cache_dir=args.cache_dir
        # cnt=0
        # for d in tqdm(dataloader):
        #     temp_result=generate(d,clm_mode=clm_mode,mono=mono)
        
        #     with open(args.cache_dir+f'/{cnt}','wb') as o:
        #         pickle.dump(temp_result,o,protocol=5,)
        #     cnt+=1
        # exit()
            
        aa=pickle.load(open('/data/ruanjh/best_training_method/t5/real_total_dict_train_supervised','rb'))
        print(aa==real_total_clm_dict)
        import pdb
        pdb.set_trace()
        # print('被平滑的token数和总token数',multiple_cnt, total_len)
        # print(multiple_cnt / total_len)
        print('real_total_dict的元素数',len(real_total_dict))
        output_dir = f"/data/ruanjh/best_training_method/t5/real_total_dict_{output}"
        pickle.dump(real_total_dict, open(output_dir+'_supervised', "wb"),protocol=5,)
        if clm_mode:
            pickle.dump(real_total_clm_dict,open(output_dir+'_clm', "wb"),protocol=5,)
        print(f"文件被写入{output_dir}")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# print('saved')

# print([total_dict[x] for x in total_dict.keys()])


# import pickle


# from tqdm import tqdm


# def get_data(rank,total_dict,tokenizer,zero_prob):
#     data=[]

#     for x in tqdm(total_dict):
#         temp_dict={}
#         temp_dict['target']=transform_to_log_prob(torch.tensor(x[1]),len(tokenizer),zero_prob)
#         temp_dict['input_ids']=x[0][0][1]
#         temp_dict['labels']=x[0][1][1]
#         data.append(temp_dict)
#     with open(f'/data/ruanjh/best_training_method/special_iwslt17/train.{str(zero_prob)[2:]}','a',) as o:
#         for d in data:
#             s=json.dumps(d,ensure_ascii=False)
#             o.write(s+'\n')
#     # dict[f'{rank}']=data

# def split_list(lst, n):

#     avg = len(lst) / float(n)
#     return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

# if __name__ == "__main__":

#     tokenizer=BartTokenizerFast.from_pretrained('/data/ruanjh/best_training_method/bart')
#     with open('/data/ruanjh/best_training_method/total_dict','rb') as f:
#         total_dict=pickle.load(f)

#     zero_prob=0.25
#     start_time = time.time()
#     world_size=32
#     # 定义一个列表用于保存进程
#     processes = []
#     results = []
#     # dict = mp.Manager().dict()
#     data_all = [data_sample for data_sample in total_dict.items()]

#     data_subsets = split_list(data_all,world_size)

#     # 创建并启动多个进程
#     for i in range(world_size):  # 假设有4个核心，可以同时执行4个进程
#         process = mp.Process(target=get_data,args=(i,data_subsets[i],tokenizer,zero_prob))
#         process.start()
#         processes.append(process)


#     # 等待所有进程执行完成
#     for process in processes:
#         process.join()

#     # real_data=[]

#     # for i in range(world_size):
#     #     real_data+=dict[f'{i}']


#     # print(len(real_data))
#     print("Time taken with processes:", time.time() - start_time)
