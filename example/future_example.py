# coding=utf-8
# Created by Meteorix at 2019/7/22

import time
from tqdm import tqdm
from service_streamer import ThreadedStreamer, Streamer, RedisStreamer
from example.bert_model import TextInfillingModel, ManagedBertModel


def main():
    batch_size = 64
    model = TextInfillingModel()
    # streamer = ThreadedStreamer(model.predict, batch_size=batch_size, max_latency=0.1)
    streamer = Streamer(ManagedBertModel, batch_size=batch_size, max_latency=0.1, worker_num=4, cuda_devices=(0, 1, 2, 3))
    streamer._wait_for_worker_ready()
    # streamer = RedisStreamer()

    text = "Happy birthday to [MASK]"
    num_epochs = 100
    total_steps = batch_size * num_epochs

    t_start = time.time()
    for i in tqdm(range(num_epochs)):
        output = model.predict([text])
    t_end = time.time()
    print('model prediction time', t_end - t_start)

    t_start = time.time()
    for i in tqdm(range(num_epochs)):
        output = model.predict([text] * batch_size)
    t_end = time.time()
    print('[batched]sentences per second', total_steps / (t_end - t_start))

    t_start = time.time()
    xs = []
    for i in range(total_steps):
        future = streamer.submit([text])
        xs.append(future)

    for future in tqdm(xs):  # 先拿到所有future对象，再等待异步返回
        output = future.result(timeout=20)
    t_end = time.time()
    print('[streamed]sentences per second', total_steps / (t_end - t_start))

    streamer.destroy_workers()
    time.sleep(10)


if __name__ == '__main__':
    main()
