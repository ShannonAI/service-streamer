# -*- coding: utf-8 -*-

# @File    : dummy_model.py
# @Date    : 2021-05-11
# @Author  : skym

# coding=utf-8
# Created by Meteorix at 2019/7/30
import logging
import time
import torch
from typing import List
from pytorch_transformers import *
from service_streamer import ManagedModel

logging.basicConfig(level=logging.ERROR)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



class DummyModel(object):
    def __init__(self, name="dummy"):
        self.name = name

    def predict(self, batch: List[str]) -> List[str]:
        return [f'{item}_{self.name}' for item in batch]


class ManagedDummyModel(ManagedModel):

    def init_model(self):
        self.model = DummyModel()
        time.sleep(3)

    def predict(self, batch):
        return self.model.predict(batch)


if __name__ == "__main__":
    batch = ["twinkle twinkle [MASK] star.",
             "Happy birthday to [MASK].",
             'the answer to life, the [MASK], and everything.']
    model = DummyModel()
    outputs = model.predict(batch)
    print(outputs)
