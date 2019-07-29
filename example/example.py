# coding=utf-8
# Created by Meteorix at 2019/7/22

import torch
import time
import os
from pytorch_transformers import *
from service_streamer import ThreadedStreamer as Streamer
from service_streamer import StreamWorker, GpuWorkerManager


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


def main():
    model = Model()
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    streamer = Streamer(model.predict, 1000, 0.1)
    num_times = 1000

    class GpuWorkers(GpuWorkerManager):

        def gpu_worker(self, index, gpu_num):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index % gpu_num)
            StreamWorker(model.predict, 64, max_latency=0.1).run_forever()

    t_start = time.time()
    for i in range(num_times):
        output = model.predict([text])
    t_end = time.time()
    print('model prediction time', t_end - t_start)

    t_start = time.time()
    output = model.predict([text] * num_times)
    t_end = time.time()
    print('model batch prediction time', t_end - t_start)

    t_start = time.time()
    xs = []
    for i in range(num_times):
        future = streamer.submit([text])
        xs.append(future)
    for future in xs:  # 先拿到所有future对象，再等待异步返回
        output = future.result()
    t_end = time.time()
    print('streamer prediction time', t_end - t_start)

    GpuWorkers().run_workers_forever(worker_num=8, gpu_num=4)


if __name__ == '__main__':
    main()
