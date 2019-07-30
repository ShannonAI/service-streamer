# coding=utf-8
# Created by Meteorix at 2019/7/22

import torch
import time
import os
from tqdm import tqdm
from pytorch_transformers import *
from service_streamer import ThreadedStreamer, Streamer
from service_streamer import RedisWorker, GpuWorkerManager


class Model(object):
    def __init__(self):
        self.model_path = "bert-base-uncased"
        self.model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-24_H-1024_A-16"
        self.config = BertConfig.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertModel(self.config)
        self.bert.eval()
        # self.bert.to("cuda")

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
    max_batch = 128
    model = Model()
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    streamer = Streamer(model.predict, batch_size=max_batch, max_latency=0.1, worker_num=2, cuda_devices=(0, 1), model=model)
    # streamer = ThreadedStreamer(model.predict, batch_size=max_batch, max_latency=0.1)
    num_times = 3000

    class GpuWorkers(GpuWorkerManager):

        def gpu_worker(self, index, gpu_num):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index % gpu_num)
            RedisWorker(model.predict, 64, max_latency=0.1).run_forever()

    """
    t_start = time.time()
    for i in tqdm(range(num_times)):
        output = model.predict([text])
    t_end = time.time()
    print('model prediction time', t_end - t_start)
    """

    t_start = time.time()
    inputs = [text] * num_times
    for i in tqdm(range(num_times // max_batch + 1)):
        output = model.predict(inputs[i*max_batch:(i+1)*max_batch])
        print(len(output))
    t_end = time.time()
    print('model batch prediction time', t_end - t_start)

    t_start = time.time()
    xs = []
    for i in range(num_times):
        future = streamer.submit([text])
        xs.append(future)

    for future in tqdm(xs):  # 先拿到所有future对象，再等待异步返回
        output = future.result(timeout=20)
    t_end = time.time()
    print('streamer prediction time', t_end - t_start)

    # streamer._worker_process.join()
    # GpuWorkers().run_workers_forever(worker_num=8, gpu_num=4)


if __name__ == '__main__':
    main()

