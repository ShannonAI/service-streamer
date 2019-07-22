# coding=utf-8
# Created by Meteorix at 2019/7/22

import torch
import time
from pytorch_transformers import *


model_path = "bert-base-uncased"
model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"
config = BertConfig.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel(config)

input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])

s = time.time()
for i in range(10):
    batch = [input_ids] * 100
    res = model(input_ids)
    print(res)

print(time.time() -s)
