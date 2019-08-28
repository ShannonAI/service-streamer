# coding=utf-8
from multiprocessing import freeze_support
from service_streamer import run_redis_workers_forever
from example.bert_model import ManagedBertModel
from tests.test_service_streamer import ManagedVisionModel

if __name__ == "__main__":
    freeze_support()
    run_redis_workers_forever(ManagedVisionModel, 64, 0.1, worker_num=4, cuda_devices=(1,), prefix='test_1')