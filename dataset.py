from torch.utils.data import Dataset
import torch


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

def _make_causal_mask(
    input_ids_shape: torch.Size, past_key_values_length: int = 0,dtype: torch.dtype=torch.int64
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # import pdb
    # pdb.set_trace()
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.iinfo(dtype).min,)
    mask_cond = torch.arange(mask.size(-1),)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len).clone()


def prepare_decoder_attention_mask( attention_mask, input_shape,): 
    
     # create causal mask 
     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len] 
    combined_attention_mask = None 
    if input_shape[-1] > 1: 
        combined_attention_mask = _make_causal_mask( 
            input_shape, 
        )
    return combined_attention_mask

def transform_to_log_prob(knns, vocab_size, zero_prob):
    if len(knns)==0:
        # return torch.zeros((vocab_size))
        return None
    vocab_tensor = torch.bincount(knns, minlength=vocab_size)

    # print(vocab_tensor)
    def fun(x):
        non_zero_index = torch.nonzero(vocab_tensor)
        zero_count = torch.eq(vocab_tensor, 0).sum().item()

        sum = 0
        for index in non_zero_index:
            sum += torch.exp(vocab_tensor[index] / x)
        return zero_count / (zero_count + sum) - zero_prob
    if zero_prob==0:
        knn_temperature=0.000001 # 要放大，才能压低概率
    elif zero_prob==1:
        knn_temperature=1 # 其实是我留的特殊情况，因为不可能给非0区分1的。这个情况就是靠softmax自己平滑就好了。
    else:
        knn_temperature,info, status, message = fsolve(fun, 0.1,full_output=True)
        start_point=10
        while status in [2, 3, 4, 5]:
            import pdb
            pdb.set_trace()
            knn_temperature,info, status, message = fsolve(fun, start_point,full_output=True)
            start_point+=1
            if start_point>50:
                break
 
    probs = torch.nn.functional.softmax(vocab_tensor / knn_temperature, dim=-1)
    # print(probs)
    return probs


def get_data(x, tokenizer, zero_prob,hybrid):

    # print(x)
    # print(x[1])

    temp_dict = {}
    temp_dict["input_ids"] = x[0][0]
    temp_dict["decoder_input_ids"] = x[0][1]
    # 1.0版本的
    # temp_dict["target"] = transform_to_log_prob(
    #     torch.tensor(x[1]), len(tokenizer), zero_prob
    # )

    # 2.0版本的：
    # for xx in x[1]:
    #     if len(xx)>1:
    #         print('xx',xx)
    #         print('?')
    #         # print(len(x[1]))
    #         exit()
    # all_prob=[transform_to_log_prob(
    #     torch.tensor(xx), len(tokenizer), zero_prob
    # )  for xx in x[1]] # 这里有一点点麻烦，xx很可能是空的
    # # NOTE seq_len * vocab_size
    # temp_dict["target"] = torch.stack(all_prob)
    
    # print(x[1])
    # 3.0版本
    # BUG 记得把这里改回去，t5神金的，embedding layer维度和len(tokenizer)不一样。。
    # 可能得考虑读取config.json里的vocab_size

    all_prob=[transform_to_log_prob(
        torch.tensor(list(xx.elements())), 32128, zero_prob
    )  for xx in x[1]] # 这里有一点点麻烦，xx很可能是空的
    # NOTE seq_len * vocab_size
    temp_dict["target"] = torch.stack(all_prob)
    # assert temp_dict["target"].size(0)==len(temp_dict["decoder_input_ids"]),f'{x},{temp_dict["target"].size(0)},{len(temp_dict["decoder_input_ids"])}'
    # print(temp_dict["decoder_input_ids"].size(0))
    # assert ==,f"fuck batch is {batch},logit_index is{logit_index},{torch.cat(target).size(0)}"
       

    return temp_dict


class SpecialDataset(Dataset):
    def __init__(self, data, tokenizer, zero_prob):

        self.data = [data_sample for data_sample in data.items()]
        self.zero_prob = zero_prob

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return get_data(
            self.data[index], tokenizer=self.tokenizer, zero_prob=self.zero_prob
        )

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


class SpecialDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> torch.Any:
        
        target = [d["target"] for d in batch]
        input_ids = [list(d["input_ids"]) for d in batch]
        decoder_input_ids = [list(d["decoder_input_ids"]) for d in batch]
        logit_index = [len(x)-1 for x in decoder_input_ids] # 不需要预测最后一个词？
        
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

        # decoder_attention_mask = prepare_decoder_attention_mask(
        #     decoder_input_ids.attention_mask,decoder_input_ids.input_ids.shape
        # )
        # print(decoder_input_ids.input_ids.shape)
        # print(decoder_attention_mask.shape)
        # assert torch.cat(target).size(0)==sum(logit_index)+len(logit_index),f"fuck batch is {batch},logit_index is{logit_index},{torch.cat(target).size(0)}"
        return {
            "target": torch.cat(target),  
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "decoder_input_ids": decoder_input_ids.input_ids, # 这里有mask,怎么办呢?
            # "decoder_attention_mask" :decoder_attention_mask,
            "logit_index" : logit_index,
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



# ------------------------------------

class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        input_id = tokenizer(src,add_special_tokens=False)
        
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
    
# -------------------------------------------------------------------------
class PreprocessDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        tgt = [d["de"] for d in data]
        input_id = tokenizer(src,add_special_tokens=False)
        tgt_id = tokenizer(tgt,add_special_tokens=False)
        # 开头就是不需要加EOS的，我试了，会让模型训完生成全空
        # 因为他学到基于第一个id永远生成EOS。
        result_matrix =[[tokenizer.eos_token_id]+ row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
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
