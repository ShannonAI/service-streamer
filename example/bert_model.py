# coding=utf-8
# Created by Meteorix at 2019/7/30
import torch
from pytorch_transformers import *
from service_streamer import ManagedModel


class Model(object):
    def __init__(self):
        self.model_path = "bert-base-uncased"
        self.model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-24_H-1024_A-16"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertForMaskedLM.from_pretrained(self.model_path)
        self.bert.eval()
        self.bert.to("cuda")

    def predict(self, batch):
        """predict next word"""
        batch_inputs = []
        masked_indexes = []

        # add token cls & mask
        for text in batch:
            tokenized_text = self.tokenizer.tokenize(text)
            tokenized_text.insert(0, "[CLS]")
            tokenized_text.append("[MASK]")
            length = len(tokenized_text)
            masked_indexes.append(length-1)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # print(tokenized_text, indexed_tokens)
            batch_inputs.append(indexed_tokens)

        # padding to same length
        max_len = max([len(tmp) for tmp in batch_inputs])
        pad_inputs = []
        for tmp_sent in batch_inputs:
            tmp_sent.extend([0] * (max_len - len(tmp_sent)))
            pad_inputs.append(tmp_sent)

        tokens_tensor = torch.tensor(pad_inputs).to("cuda")

        with torch.no_grad():
            predictions = self.bert(tokens_tensor)[0]

        batch_outputs = []
        for i in range(len(batch_inputs)):
            predicted_index = torch.argmax(predictions[i, masked_indexes[i]]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            batch_outputs.append(predicted_token)

        return batch_outputs


class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = Model()

    def predict(self, batch):
        return self.model.predict(batch)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    m = Model()
    outputs = m.predict(["Today is your lucky", "Happy birthday to"])
    print(outputs)
