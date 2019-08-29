# coding=utf-8
from multiprocessing import freeze_support
from service_streamer import run_redis_workers_forever
from example.bert_model import ManagedBertModel
from tests.test_service_streamer import ManagedVisionModel, ManagedVisionModel2
import threading


# def init_redisworkers():
#     thread1 = threading.Thread(target=run_redis_workers_forever,args=(ManagedVisionModel, 64, 0.1, 4, (1,), "localhost:6379", 'test'))
#     thread2 = threading.Thread(target=run_redis_workers_forever,args=(ManagedVisionModel, 64, 0.1, 4, (1,), "localhost:6379", 'test_1'))
#     thread1.start()
#     thread2.start()

# if __name__ == "__main__":
#     freeze_support()
#     init_redisworkers()
#     # run_redis_workers_forever(ManagedVisionModel, 64, 0.1, worker_num=4, cuda_devices=(1,), prefix='test_1')

thread1 = threading.Thread(target=run_redis_workers_forever,args=(ManagedVisionModel, 64, 0.1, 4, (1,), "localhost:6379", 'test'))
thread2 = threading.Thread(target=run_redis_workers_forever,args=(ManagedVisionModel2, 64, 0.1, 4, (1,), "localhost:6379", 'test_1'))
thread1.start()
thread2.start()
