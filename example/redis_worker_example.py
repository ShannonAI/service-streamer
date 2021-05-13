# coding=utf-8
# Created by Meteorix at 2019/7/30
from service_streamer import run_redis_workers_forever
from dummy_model import ManagedDummyModel


if __name__ == "__main__":
    run_redis_workers_forever(ManagedDummyModel, 64, 0.1, worker_num=1, cuda_devices=None, max_wait_time=10)
