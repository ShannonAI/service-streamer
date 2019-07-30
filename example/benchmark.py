# coding=utf-8
# Created by Meteorix at 2019/7/30
import time
import requests
from multiprocessing.pool import ThreadPool


def send_naive_request(index):
    r = requests.post("http://localhost:5005/naive", data={"s": ["happy birthday to"]})
    assert r.status_code == 200


def send_stream_request(index):
    r = requests.post("http://localhost:5005/stream", data={"s": ["Happy birthday to"]})
    assert r.status_code == 200


def bench_request(total_num=400, concurrency=64):
    pool = ThreadPool(concurrency)

    start = time.time()
    pool.map(send_naive_request, range(total_num))
    cost = time.time() - start

    print(f"[naive]{total_num/cost} sentences per second")

    start = time.time()
    pool.map(send_stream_request, range(total_num))
    cost = time.time() - start

    print(f"[stream]{total_num/cost} sentences per second")


if __name__ == "__main__":
    bench_request()
