from torch import bfloat16
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset,DataLoader

from transformers import AutoTokenizer,DefaultDataCollator,GenerationConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefaultDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data=tokenizer(data,return_tensors='pt',padding=True)
        import pdb
        pdb.set_trace()
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)





class NiuInference:
    def __init__(self,model_dir,data,dtype=bfloat16,dataset=None,data_collator=None,output_path='niuinference.out',auto_batch_size=True,batch_size=1,generation_config=None):
        self.model_dir=model_dir
        self.dtype=dtype
        self.data=data
        self.dataset=dataset
        self.data_collator=data_collator
        self.output_path=output_path
        self.batch_size=batch_size
        self.auto_batch_size=auto_batch_size
        self.generation_config=generation_config
        
        
    def _load_model_and_tokenizer(self,device):
        # BUG 这里也需要改
        model=AutoModelForSeq2SeqLM.from_pretrained(self.model_dir,torch_dtype=self.dtype,low_cpu_mem_usage=True)
        model.to(device)
        tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        return model,tokenizer

    def get_pred(self,rank,out_path,data,dict):
        try:
            device = torch.device(f'cuda:{rank}')
            model, tokenizer = load_model_and_tokenizer(device)
            # print(tokenizer.special_token)
            # 这里直接用mydataset没关系，因为这是t5类型，decoder那边会自己给个[EOS]开头吧
            if self.dataset is not None:
                dataset=self.dataset(data=data,tokenizer=tokenizer)
            else:
                dataset=DefaultDataset(data=data,tokenizer=tokenizer)
            
            if self.data_collator is not None:
                collator=self.data_collator(tokenizer,model=model,padding=True)
            else:
                collator= DefaultDataCollator(tokenizer,model=model,padding=True)
            dataloader=DataLoader(dataset,2,collate_fn=collator,pin_memory=True,num_workers=4)
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
        except Exception as e:
            logger.error(f'rank {rank}的推理过程解析出错了')
            logger.error(e)
            
            
    
    def split_list(self,lst, n):
        avg = len(lst) / float(n)
        return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

    def run(self,):
        
        if self.auto_batch_size:
            bsz=_get_batch_size()
        world_size = torch.cuda.device_count()
        mp.set_start_method('spawn', force=True)
        data_subsets = split_list(self.data,world_size)
        processes = []
        manager = mp.Manager()
        record_dict = manager.dict()
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank,self.output_path,data_subsets[rank],record_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        with open(out_path, "w", encoding="utf-8") as f:
            for rank in range(world_size):
                for r in record_dict[f'{rank}']:
                    f.write(r.replace('\n','\\n')+'\n')


    def _get_batch_size(
        self,
        model: nn.Module,
        device: torch.device,
        input_shape: t.Tuple[int, int, int],
        output_shape: t.Tuple[int],
        dataset_size: int,
        max_batch_size: int = None,
        num_iterations: int = 5,
    ) -> int:
        model,tokenizer=self._load_model_and_tokenizer(torch.device('cuda:0'))
        if self.generation_config is not None:
            if generation_config.max_position_embeddings is not None:
                length=max_input_length_from_dataset+generation_config.max_position_embeddings
            elif generation_config.max_length is not None:
                length=generation_config.max_length
        else:
            length=model.config.max_position_embeddings
        
        

        batch_size = 2
        while True:
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
                break
            if batch_size >= dataset_size:
                batch_size = batch_size // 2
                break
            try:
                for _ in range(num_iterations):
                    # dummy inputs and targets
                    inputs = torch.rand(*(batch_size, *input_shape), device=device)
                    targets = torch.rand(*(batch_size, *output_shape), device=device)
                    outputs = model(inputs)
                    loss = F.mse_loss(targets, outputs)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                batch_size *= 2
            except RuntimeError:
                batch_size //= 2
                break
        del model, optimizer
        torch.cuda.empty_cache()
        import pdb
        pdb.set_trace()
        return batch_size
    

i=NiuInference(model_dir='/data/ruanjh/best_training_method/gemma-2b',data=['你好','我爱你','啊啊啊啊啊啊啊啊啊啊啊啊啊啊'])
i.run()
