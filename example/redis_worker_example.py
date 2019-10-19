# coding=utf-8
# Created by Meteorix at 2019/7/30
from service_streamer import run_redis_workers_forever
from bert_model import ManagedBertModel


if __name__ == "__main__":
    run_redis_workers_forever(ManagedBertModel, 64, 0.1, worker_num=4, cuda_devices=(0, 1, 2, 3))
