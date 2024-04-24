from torch.utils.data import Dataset
import torch
import time

class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        tgt = [d["de"] for d in data]
        input_id = tokenizer(src,add_special_tokens=False)
        tgt_id = tokenizer(tgt,add_special_tokens=False)
        # 开头就是不需要加EOS的，我试了，会让模型训完生成全空
        # 因为他学到基于第一个id永远生成EOS。
        # NOTE 但是为啥第二次训练掉这么多性能，更新步数吗？
        result_matrix = [ row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
        self.data = {
            "input_ids": input_id.input_ids,
            # "attention_mask": input_id.attention_mask,
            "labels": result_matrix,
            # "decoder_attention_mask": tgt_id.attention_mask,
        }

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "labels": self.data["labels"][index],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data["input_ids"])


# ----------------------------------------------------------------------------------------------
from scipy.optimize import fsolve
from sympy import Eq, solve,symbols,Function


def speedup_map(length):
    if length<10:
        return 0.1
    elif length<100:
        return 0.2
    elif length<500:
        return 10
    elif length<1000:
        return 50
    elif length<10000:
        return 130
    elif length<20000:
        return 265
    return 1000

def transform_to_log_prob(knns, vocab_size=None, zero_prob=None,quick_mode=True,):

    if len(knns)==0:
        # return torch.zeros((vocab_size))
        return None
    else:
        # 不需要拟合温度的情况。
        if zero_prob==0:
            knn_temperature=0.000001 # 要放大，才能压低概率
        else:
            if not quick_mode:
                # values,_=torch.topk(knns,10)
                # x=torch.sum(values,dim=-1)
                # knn_temperature = 0.11*(x**0.76)
                # probs = torch.nn.functional.softmax(torch.div(knns,knn_temperature.unsqueeze(1)), dim=-1)
                
                vocab_tensor = torch.bincount(knns, minlength=vocab_size)
                non_zero_index = torch.nonzero(vocab_tensor)
                zero_count = torch.eq(vocab_tensor, 0).sum().item()
                def fun(x):
                    if x==0:
                        return 10000
                    
                    # sum = 0
                    # for index in non_zero_index:
                    #     sum += torch.exp(vocab_tensor[index] / x)
                    tensor_with_temperature=vocab_tensor/x
                    exp_tensor = torch.exp(tensor_with_temperature)
                    # nonzero_exp_tensor=exp_tensor[non_zero_index[:,0],non_zero_index[:,1]]
                    sum_exp=torch.sum(exp_tensor) # 注意，这个地方因为0部分的顶上e之后是1，其实等价于包含了zero_count的，不需要特地分开求
                    
                    return zero_count / (sum_exp) - zero_prob
                
                knn_temperature,info, status, message = fsolve(fun, 0.08,full_output=True)
                start_point=10
                while status in [2, 3, 4, 5]:
                    knn_temperature,info, status, message = fsolve(fun, start_point,full_output=True)
                    start_point+=1
                    if start_point>50:
                        print('失败了，没找到温度')
                        break
            else:
                # 预先计算,免得在多次fun的迭代里都要重算
                zero_count_per_tensor = torch.sum(knns == 0, dim=-1)
                non_zero_index = torch.nonzero(knns)
                # 分母
                bsz=knns.size(0)
                
                def jacobian(x):
                    # 自己写的呀科比矩阵比他自动推导的还慢，流汗黄豆了家人们
                    tensor_with_temperature=knns/x
                    exp_tensor = torch.exp(tensor_with_temperature) # e^xi/T 
                    special_tensor=knns/(x**2) # xi/T^2 (本来这里应该有个负号的，但是被分子m之前的负号抵消了)
                    factor=torch.mul(exp_tensor, special_tensor).sum(dim=1)*zero_count_per_tensor
                    sum_exp = torch.sum(exp_tensor,dim=-1)
                    square_sum=torch.square(sum_exp)
                    return [torch.sum(sum_exp/square_sum)]
                    
                def fun(x):
                    if x<=0:
                        return - 10000
                    # import pdb
                    # pdb.set_trace()
                    tensor_with_temperature=knns/x
                    
                    exp_tensor = torch.exp(tensor_with_temperature)
                    # nonzero_exp_tensor=exp_tensor[non_zero_index[:,0],non_zero_index[:,1]]
                    # sum_exp = torch.sum(nonzero_exp_tensor,dim=-1)
                    sum_exp = torch.sum(exp_tensor,dim=-1)
                    result=torch.sum(zero_count_per_tensor/sum_exp)  - bsz*zero_prob
                    return result

                # print(knns.nonzero().size(0))
                record=False

                knn_temperature,info, status, message = fsolve(fun, 0.07,full_output=True,col_deriv=True,factor=10)
                # x=symbols('x')
                # f=Function(fun)
                # knn_temperature = solve(Eq(fun(x), 0),x )
                # 2\32\28\3\0.85\4.59\0.51\1.49\49\1.68
                
                if status in [2, 3, 4, 5]:
                    knn_temperature,info, status, message = fsolve(fun, 500,full_output=True,col_deriv=True,)
                
                if status in [2, 3, 4, 5]:
                    knn_temperature,info, status, message = fsolve(fun, 0.03,full_output=True,col_deriv=True)
                
                if status in [2, 3, 4, 5]:
                    initial_start_point = 1
                    start_point = initial_start_point
                    end_point=50
                    interval = max((end_point-start_point)//5,1)
                
                    while status in [2, 3, 4, 5]:
                        knn_temperature,info, status, message = fsolve(fun, start_point,full_output=True,col_deriv=True)
                        # print(start_point,interval,initial_start_point,end_point,knn_temperature)
                        
                        start_point+=interval
                        if start_point>end_point:
                            print(start_point,info,knn_temperature)
                            if knn_temperature<0:
                                knn_temperature=1
                            # import pdb
                            # pdb.set_trace() 
                            print('失败了，没找到温度')
                            break
            # print(knn_temperature)
            probs = torch.nn.functional.softmax(knns / knn_temperature, dim=-1)
            # import pdb
            # pdb.set_trace()
        
    return probs


def get_data(supervised, tokenizer, zero_prob,clm,hybrid,quick_mode,div_mode):

    # print(x)
    # print(x[1])
    
    temp_dict = {}
    temp_dict["input_ids"] = supervised[0][0]
    temp_dict["decoder_input_ids"] = supervised[0][1]


    # print(x[1])
    # 3.0版本
    # BUG 记得把这里改回去，t5神金的，embedding layer维度和len(tokenizer)不一样。。
    # 可能得考虑读取config.json里的vocab_size

    if div_mode:
        # print('div_mode')
        x=torch.stack([torch.bincount(torch.tensor(list(xx.elements())), minlength=32128) for xx in supervised[1]])
        all_prob_supervised=x/torch.sum(x,dim=-1,keepdim=True)
        # if (torch.sum(x,dim=-1,keepdim=True)>1).any():
        
        
        #     import pdb
        #     pdb.set_trace()
        if clm is not None:
            x=torch.stack([torch.bincount(torch.tensor(list(xx.elements())), minlength=32128) for xx in clm[1]])
            all_prob_clm=x/torch.sum(x,dim=-1,keepdim=True)
        # if clm!=supervised:
        #     import pdb
        #     pdb.set_trace()
        
    else:
        if quick_mode:
            x=torch.stack([torch.bincount(torch.tensor(list(xx.elements())), minlength=32128) for xx in supervised[1]])
            all_prob_supervised=transform_to_log_prob(x,zero_prob=zero_prob,quick_mode=quick_mode)

            try:
                x=torch.stack([torch.bincount(torch.tensor(list(xx.elements())), minlength=32128) for xx in clm[1]])
                all_prob_clm=transform_to_log_prob(x,zero_prob=zero_prob,quick_mode=quick_mode)
            except:
                print(supervised)
                print(clm)
                print([list(xx.elements()) for xx in clm[1]])
                print([torch.tensor(list(xx.elements())) for xx in clm[1]])
            
        else:
            #--------------------------------------------------------------------------------------------------------
            all_prob_supervised=torch.stack([transform_to_log_prob(
                torch.tensor(list(xx.elements())), 32128, zero_prob,quick_mode=quick_mode
            )  for xx in supervised[1]]) # 这里有一点点麻烦，xx很可能是空的
            all_prob_clm=torch.stack([transform_to_log_prob(
                torch.tensor(list(xx.elements())), 32128, zero_prob,quick_mode=quick_mode
            )  for xx in clm[1]]) # 这里有一点点麻烦，xx很可能是空的

    # print('s time',time2-time1)
    # print('c time',time3-time2)
    # all_prob_supervised=torch.stack(all_prob_supervised)
    # all_prob_clm=torch.stack(all_prob_clm)

    temp_dict["target"] =0.8*all_prob_supervised+0.2*all_prob_clm
    # temp_dict["target"] = all_prob_supervised
    temp_dict["debug_target"] = list(supervised[0][1])[1:]+[1]
    # import pdb
    # pdb.set_trace() # [torch.sum(xx[0]) for xx in [torch.topk(x,l) for x,l in zip(all_prob_clm,[len(x) for x in clm[1]])]]
    # NOTE seq_len * vocab_size
    # temp_dict["target"] = torch.stack(all_prob)
    # assert temp_dict["target"].size(0)==len(temp_dict["decoder_input_ids"]),f'{x},{temp_dict["target"].size(0)},{len(temp_dict["decoder_input_ids"])}'
    # print(temp_dict["decoder_input_ids"].size(0))
    # assert ==,f"fuck batch is {batch},logit_index is{logit_index},{torch.cat(target).size(0)}"

    return temp_dict


class SpecialDataset(Dataset):
    def __init__(self, supervised, tokenizer=None, zero_prob=0,clm=None,hybrid=False,quick_mode=True,div_mode=False):

        self.supervised = [data_sample for data_sample in supervised.items()]
        if clm is not None:
            self.clm = [data_sample for data_sample in clm.items()]
        else:
            self.clm = None
        self.zero_prob = zero_prob

        self.tokenizer = tokenizer
        self.hybrid=hybrid
        self.quick_mode=quick_mode
        self.div_mode=div_mode

    def __getitem__(self, index):
        if self.clm:
            return get_data(
                self.supervised[index], tokenizer=self.tokenizer, zero_prob=self.zero_prob,clm=self.clm[index],hybrid=self.hybrid,
                quick_mode=self.quick_mode,div_mode=self.div_mode,
            )
        else:
            return get_data(
                self.supervised[index], tokenizer=self.tokenizer, zero_prob=self.zero_prob,clm=None,hybrid=self.hybrid,
                quick_mode=self.quick_mode,div_mode=self.div_mode,
            )

    def __len__(self):
        return len(self.supervised)


from torch.nn.utils.rnn import pad_sequence


class SpecialDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> torch.Any:
        
        target = [d["target"] for d in batch]
        input_ids = [list(d["input_ids"]) for d in batch]
        decoder_input_ids = [list(d["decoder_input_ids"]) for d in batch]
        logit_index = [len(x)-1 for x in decoder_input_ids] # 不需要预测最后一个词？
        debug_target = [d["debug_target"]  for d in batch]
        # print("input_ids", input_ids)
        # print("decoder_input_ids", decoder_input_ids)
        # print("logit_index", logit_index)
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        # import pdb
        # pdb.set_trace()
        # print(input_ids.attention_mask.dtype)
        decoder_input_ids = self.tokenizer.pad(
            {"input_ids": decoder_input_ids}, return_tensors="pt", padding=True
        )
        
        max_length = max(map(len, debug_target))
        # 补齐列表并转换为PyTorch张量
        debug_target = [sublist + [-100] * (max_length - len(sublist)) for sublist in debug_target]
        debug_target = torch.tensor(debug_target)
        
        # decoder_attention_mask = prepare_decoder_attention_mask(
        #     decoder_input_ids.attention_mask,decoder_input_ids.input_ids.shape
        # )
        # print(decoder_input_ids.input_ids.shape)
        # print(decoder_attention_mask.shape)
        # assert torch.cat(target).size(0)==sum(logit_index)+len(logit_index),f"fuck batch is {batch},logit_index is{logit_index},{torch.cat(target).size(0)}"
        # print('done')
        return {
            "target": torch.cat(target),  
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "decoder_input_ids": decoder_input_ids.input_ids, # 这里有mask,怎么办呢?
            # "decoder_attention_mask" :decoder_attention_mask,
            "logit_index" : logit_index,
            "debug_target":debug_target
        }
        input_ids = [x["input_ids"] for x in batch]
        input_ids = self.tokenizer.pad(
            input_ids,
            return_tensors="pt",
        )
        decoder_input_ids = [x["decoder_input_ids"] for x in batch]
        decoder_input_ids = self.tokenizer.pad(
            decoder_input_ids,
            return_tensors="pt",
        )
        return {
            "input_ids": input_ids.input_ids,
            "attentio_mask": input_ids.attention_mask,
            "decoder_input_ids": decoder_input_ids.input_ids,
            "target": [x["target"] for x in batch],
        }


