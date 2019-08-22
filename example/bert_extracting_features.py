# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
from pytorch_transformers.modeling_bert import BertModel
from service_streamer import ManagedModel
import torch
from typing import List

logging.basicConfig(level=logging.ERROR)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
MODEL_DIR = '/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12/'
input_ids = [101, 2572, 3217, 5831, 5496, 2010, 2567, 1010, 3183, 2002, 2170, 1000, 1996, 7409, 1000, 1010, 1997, 9969,
             4487, 23809, 3436, 2010, 3350, 1012, 102, 7727, 2000, 2032, 2004, 2069, 1000, 1996, 7409, 1000, 1010, 2572,
             3217, 5831, 5496, 2010, 2567, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0]
input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
input_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
unique_ids = 1
batch = [{'input_ids': input_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids, 'unique_ids': unique_ids}]


class BertExtractor(object):
    def __init__(self):
        self.model_dir = MODEL_DIR
        self.bert_model = BertModel.from_pretrained(MODEL_DIR).cuda()

    def predict(self, batch):
        batch_input_ids: List[List[int]] = [instance['input_ids'] for instance in batch]
        batch_attention_mask: List[List[int]] = [instance['input_mask'] for instance in batch]
        batch_token_type_ids: List[List[int]] = [instance['input_type_ids'] for instance in batch]

        outputs = self.bert_model.forward(input_ids=torch.LongTensor(batch_input_ids).cuda(),
                                          attention_mask=torch.LongTensor(batch_attention_mask).cuda(),
                                          token_type_ids=torch.LongTensor(batch_token_type_ids).cuda()
                                          )
        features = outputs[0].data.tolist()
        return [{'features': feature, 'unique_ids': instance['unique_ids']} for feature, instance in zip(features, batch)]


class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = BertExtractor()

    def predict(self, batch):
        return self.model.predict(batch)


def main():
    model = BertExtractor()
    outputs = model.predict(batch)
    print(outputs)


if __name__ == "__main__":
    main()
