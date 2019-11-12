# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
import torch
from typing import List
from pytorch_transformers import *
from service_streamer import ManagedModel

logging.basicConfig(level=logging.ERROR)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class TextInfillingModel(object):
    def __init__(self, max_sent_len=16, model_path="bert-base-uncased"):
        self.model_path = model_path
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


if __name__ == "__main__":
    batch = ["twinkle twinkle [MASK] star.",
             "Happy birthday to [MASK].",
             'the answer to life, the [MASK], and everything.']
    model = TextInfillingModel()
    outputs = model.predict(batch)
    print(outputs)
