# coding=utf-8
# Created by Meteorix at 2019/8/16
import os
import threading

from vision_case.model import VisionDensenetModel, VisionResNetModel, DIR_PATH

from service_streamer import ThreadedStreamer, ManagedModel, Streamer, RedisStreamer, RedisWorker, \
    run_redis_workers_forever
import torch

BATCH_SIZE = 2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # in case ci environment do not have gpu

input_batch = []
vision_model = None
managed_model = None
single_output = None
batch_output = None
input_batch2 = None
single_output2 = None
batch_output2 = None


class ManagedVisionDensenetModel(ManagedModel):
    def init_model(self):
        self.model = VisionDensenetModel(device=device)

    def predict(self, batch):
        return self.model.batch_prediction(batch)


class ManagedVisionResNetModel(ManagedModel):
    def init_model(self):
        self.model = VisionResNetModel(device=device)

    def predict(self, batch):
        return self.model.batch_prediction(batch)


def setup_module(module):
    global input_batch, vision_model, managed_model, single_output, batch_output, input_batch2, single_output2, batch_output2

    with open(os.path.join(DIR_PATH, "cat.jpg"), 'rb') as f:
        image_bytes = f.read()
    input_batch = [image_bytes]
    vision_model = VisionDensenetModel(device=device)
    single_output = vision_model.batch_prediction(input_batch)
    batch_output = vision_model.batch_prediction(input_batch * BATCH_SIZE)

    with open(os.path.join(DIR_PATH, "dog.jpg"), 'rb') as f:
        image_bytes2 = f.read()
    input_batch2 = [image_bytes2]
    vision_model2 = VisionResNetModel(device=device)
    single_output2 = vision_model2.batch_prediction(input_batch2)
    batch_output2 = vision_model2.batch_prediction(input_batch2 * BATCH_SIZE)

    managed_model = ManagedVisionDensenetModel()
    managed_model.init_model()


def test_init_redisworkers():
    from multiprocessing import freeze_support
    freeze_support()
    thread1 = threading.Thread(target=run_redis_workers_forever, args=(
    ManagedVisionDensenetModel, 16, 0.1, 4, (0, 1, 2, 3), "localhost:6379", 'channel_for_densenet'), daemon=True)
    thread2 = threading.Thread(target=run_redis_workers_forever, args=(
    ManagedVisionResNetModel, 16, 0.1, 4, (0, 1, 2, 3), "localhost:6379", 'channel_for_resnet'), daemon=True)
    thread1.start()
    thread2.start()


def test_threaded_streamer():
    streamer = ThreadedStreamer(vision_model.batch_prediction, batch_size=8)
    single_predict = streamer.predict(input_batch)
    assert single_predict == single_output

    batch_predict = streamer.predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output


def test_managed_model():
    single_predict = managed_model.predict(input_batch)
    assert single_predict == single_output

    batch_predict = managed_model.predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output


def test_spawned_streamer():
    # Spawn releases 4 gpu worker processes
    streamer = Streamer(vision_model.batch_prediction, batch_size=8, worker_num=4, cuda_devices=(0, 1, 2, 3))
    single_predict = streamer.predict(input_batch)
    assert single_predict == single_output

    batch_predict = streamer.predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output


def test_future_api():
    streamer = ThreadedStreamer(vision_model.batch_prediction, batch_size=8)

    xs = []
    for i in range(BATCH_SIZE):
        future = streamer.submit(input_batch)
        xs.append(future)
    batch_predict = []
    # Get all instances of future object and wait for asynchronous responses.
    for future in xs:
        batch_predict.extend(future.result())
    assert batch_output == batch_predict


def test_redis_streamer():
    # Spawn releases 4 gpu worker processes
    streamer = RedisStreamer(prefix='channel_for_densenet')
    single_predict = streamer.predict(input_batch)
    assert single_predict == single_output

    batch_predict = streamer.predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output


def test_mult_channel_streamer():
    streamer_1 = RedisStreamer(prefix='channel_for_densenet')
    streamer_2 = RedisStreamer(prefix='channel_for_resnet')

    single_predict = streamer_1.predict(input_batch)
    assert single_predict == single_output

    batch_predict = streamer_1.predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output

    single_predict2 = streamer_2.predict(input_batch2)
    assert single_predict2 == single_output2

    batch_predict2 = streamer_2.predict(input_batch2 * BATCH_SIZE)
    assert batch_predict2 == batch_output2
