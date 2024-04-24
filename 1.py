from transformers import Seq2SeqTrainer,Trainer
import torch
import os
from torch.nn import KLDivLoss,CrossEntropyLoss
import torch.nn.functional as F

logit_index=[1]

model_logits=torch.rand((1,3,5))

last_logits =torch.cat([row[:index+1] for row, index in zip(model_logits, logit_index)])

target=
import pdb
pdb.set_trace()
       
if self.ce:
    ce_loss=CrossEntropyLoss()
    loss = ce_loss(last_logits,target).to(
        model_logits.device
    )
else:
    kl_loss = KLDivLoss(reduction="batchmean")
    loss = kl_loss(F.log_softmax(last_logits, dim=-1), target).to(
        model_logits.device
    )