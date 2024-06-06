import time
import torch.multiprocessing as mp
import torch

mp.set_sharing_strategy("file_system")
from collections import Counter, defaultdict

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
class PreprocessDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        tgt = [d["de"] for d in data]
        input_id = tokenizer(src,add_special_tokens=False) 
        tgt_id = tokenizer(tgt,add_special_tokens=False)
        
        # decoder 那边的输入在generate的时候会自动附着一个0作为引导生成标志
        # 至于eos,模型不学这个不知道什么时候停止
        result_matrix =[[0]+ row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
        self.data = {
            "input_ids": input_id.input_ids,
            "labels": result_matrix,
        }

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "labels": self.data["labels"][index],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data["input_ids"])


clm_mode = True
mono=False #NOTE 开mono的时候要注意 ngram和eos行为和普通clm一不一样
if __name__ == "__main__":

    file = [
        ("train", "train"),
        # ("validation", "eval"),
    ]
    mono_file='/data/ruanjh/news.2023.de.shuffled.deduped'
    # debug_cnt=0
    for input, output in file:
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/ruanjh/best_training_method/t5v1_1-large",add_special_tokens=False,
        )
        with open(f"/data/ruanjh/best_training_method/iwslt17/{input}.json") as f:
            train_data = json.load(f)

        train_dateset = PreprocessDataset(train_data, tokenizer)
        total_dict = defaultdict(Counter)
        print(len(train_data))

        if clm_mode:
            clm_dict = defaultdict(Counter)
        for j in tqdm(range(len(train_dateset)), desc="first stage"):

            d = train_dateset[j]
            # print(d)
            # d['labels']=[tokenizer.eos_token_id]+d['labels'] # 这后面不用加，因为dataset里默认是后面带了的。
            length = len(d["labels"])
            for i in range(length - 1):
                # value总是预测的下一个词
                value = d["labels"][i + 1]
                decoder_input_id = d["labels"][: i + 1]
                input_id = d["input_ids"]

                key = (tuple(input_id), tuple(decoder_input_id))
                
                total_dict[key].update([value])
                if clm_mode  and i>3: # >k就是 k+2 gram，预训练不要提供EOS的信息。      
   
                    clm_key = tuple(decoder_input_id)
                    clm_dict[clm_key].update([value])


        if mono and clm_mode and input=='train':
            print('mono file assist',mono_file)
            with open(mono_file) as f:
                cnt=0
                for line in tqdm(f,desc='assist procedure'):
                    cnt+=1
                    if cnt>500000:
                        break
                    line=line.strip()
                    assist=tokenizer(line,add_special_tokens=False,return_attention_mask=False)
                # assist = f.read().splitlines()
                    
                    assist=[0]+ assist['input_ids'] + [tokenizer.eos_token_id]
                    d = assist
                    length=len(d)
                    # print(assist)
                    for i in range(length - 1):
                        if i>2:
                            clm_key = tuple(d[: i + 1])
                            
                            clm_dict[clm_key].update([d[i + 1]])
        

        real_total_dict = defaultdict(list)
        if clm_mode:
            real_total_clm_dict= defaultdict(list)
        print("total_dict", len(total_dict))
        if clm_mode:
            print("clm_dict", len(clm_dict))
        # print('debug_cnt',debug_cnt)
        
        # 第一阶段结束 ============================
        
        multiple_cnt = 0
        total_len=0
        for j in tqdm(range(len(train_dateset)), desc="second stage"):
            d = train_dateset[j]

            decoder_input_id = d["labels"]
            input_id = d["input_ids"]
            # 这里要丢掉key里decoder_input_id最后的EOS，因为只有label需要最后是EOS，下面到length-1也是因为不需要考虑EOS位置的后续词（本来就没有）
            key = (tuple(input_id), tuple(decoder_input_id[:-1]))
            length = len(d["labels"])
            total_len+=length
 
            for i in range(length - 1):

                temp_key = (tuple(input_id), tuple(decoder_input_id[: i + 1]))
                
                
                if (
                    len(real_total_dict[key]) < len(decoder_input_id) - 1 #防止重复
                    and temp_key in total_dict
                ):
                    temp_value = total_dict[temp_key]

                    if clm_mode:

                        
                        clm_temp_key = tuple(decoder_input_id[: i + 1])
                        if clm_temp_key not in clm_dict:
                            # 如果clm这个分布是没有的，那就直接用supervised就好了。
                            # 反正是加权平均，相当于把clm这个置为0了
                            # 这个情况一般只在 大于1的ngram才会出现
                            real_total_clm_dict[key].append(temp_value)
                        else:

                            real_total_clm_dict[key].append(clm_dict[clm_temp_key])

                    
                        
                    real_total_dict[key].append(temp_value)

                else:
                    break


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
