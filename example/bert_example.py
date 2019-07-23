# coding=utf-8
# Created by Meteorix at 2019/7/22

import torch
import time
from pytorch_transformers import *


class Model(object):
    def __init__(self):
        self.model_path = "bert-base-uncased"
        self.model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-24_H-1024_A-16"
        self.config = BertConfig.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertModel(self.config)
        self.bert.eval()
        self.bert.to("cuda")

    def predict(self, batch):
        batch_inputs = []
        for text in batch:
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            batch_inputs.append(indexed_tokens)

        tokens_tensor = torch.tensor(batch_inputs).to("cuda")

        with torch.no_grad():
            encoded_layers = self.bert(tokens_tensor)[0]

        return encoded_layers.tolist()


model = Model()
text = "Who was Jim Henson ? Jim Henson was a puppeteer"

from ifluent_english.service_streamer import ThreadedStreamer as Streamer
streamer = Streamer(model.predict, 64, 0.1)

s = time.time()

# for i in range(10):
#     output = model.predict([text])
    # print(output)

# outputs = model.predict([text] * 10)

xs = []
for i in range(1000):
    future = streamer.submit([text])
    xs.append(future)

# 先拿到所有future对象，再等待异步返回
for future in xs:
    outputs = future.result()
    # print(outputs)

print(time.time() -s)
