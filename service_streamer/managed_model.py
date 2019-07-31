# coding=utf-8
# Created by Meteorix at 2019/7/22
from multiprocessing import Process
import os


class ManagedModel(object):
    def __init__(self, gpu_id):
        self.model = None
        self.gpu_id = gpu_id
        self.set_gpu_id(self.gpu_id)

    @staticmethod
    def set_gpu_id(gpu_id):
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def init_model(self):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError


class GpuWorkerManager(object):

    @staticmethod
    def gpu_worker(worker_index, gpu_num):
        devices = str(worker_index % gpu_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        print("gpu worker starting...pid: %d cuda: %s" % (os.getpid(), devices))
        # define your gpu stream worker here
        print("gpu worker exits...")

    def run_workers_forever(self, worker_num, gpu_num):
        procs = []
        for i in range(worker_num):
            p = Process(target=self.gpu_worker, args=(i, gpu_num,))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
