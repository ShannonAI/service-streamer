# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
import multiprocessing
import time
from typing import List

import torch
from pytorch_transformers import *

from service_streamer import ManagedModel, Streamer, ThreadedStreamer

logging.basicConfig(level=logging.ERROR)

multiprocessing.set_start_method("spawn", force=True)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class TextInfillingModel(object):
    def __init__(self, max_sent_len=16):
        # self.model_path = "bert-base-uncased"
        self.model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-24_H-1024_A-16"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertForMaskedLM.from_pretrained(self.model_path)
        self.bert.eval()
        self.bert.to("cuda")
        self.max_sent_len = max_sent_len

    def predict(self, batch: List[str]) -> List[str]:
        """predict masked word"""
        batch_inputs = []
        masked_indexes = []

        for text in batch:
            tokenized_text = self.tokenizer.tokenize(text)
            if len(tokenized_text) > self.max_sent_len - 2:
                tokenized_text = tokenized_text[: self.max_sent_len - 2]
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            tokenized_text += ['[PAD]'] * (self.max_sent_len - len(tokenized_text))
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            batch_inputs.append(indexed_tokens)
            masked_indexes.append(tokenized_text.index('[MASK]'))
        tokens_tensor = torch.tensor(batch_inputs).to("cuda")

        with torch.no_grad():
            # prediction_scores: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            prediction_scores = self.bert(tokens_tensor)[0]

        batch_outputs = []
        for i in range(len(batch_inputs)):
            predicted_index = torch.argmax(prediction_scores[i, masked_indexes[i]]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_index)
            batch_outputs.append(predicted_token)

        return batch_outputs


class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = TextInfillingModel()

    def predict(self, batch):
        return self.model.predict(batch)


def main():
    batch = ["twinkle twinkle [MASK] star",
             "Happy birthday to [MASK]",
             'the answer to life, the [MASK], and everything']
    model = TextInfillingModel()
    start_time = time.time()
    outputs = model.predict(batch)
    print(outputs)
    print('original model', time.time() - start_time, outputs)

    threaded_streamer = ThreadedStreamer(model.predict, 64, 0.1)
    start_time = time.time()
    outputs = threaded_streamer.predict(batch)
    print('threaded model', time.time() - start_time, outputs)

    streamer = Streamer(model.predict, 64, 0.1, worker_num=4)
    start_time = time.time()
    outputs = streamer.predict(batch)
    print('single-gpu multiprocessing', time.time() - start_time, outputs)

    managed_streamer = Streamer(ManagedBertModel, 64, 0.1, worker_num=4, cuda_devices=[0, 3])
    start_time = time.time()
    outputs = managed_streamer.predict(batch)
    print('multi-gpu multiprocessing', time.time() - start_time, outputs)

    start_time = time.time()
    xs = []
    for i in range(1):
        future = threaded_streamer.submit(batch)
        xs.append(future)
    for future in xs:
        outputs = future.result()
    print('Future API', time.time() - start_time, outputs)


if __name__ == "__main__":
    main()
