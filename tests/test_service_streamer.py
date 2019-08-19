# coding=utf-8
# Created by Meteorix at 2019/8/16
import multiprocessing
import os

from vision_case.model import VisionModel, DIR_PATH

from service_streamer import ThreadedStreamer, ManagedModel, Streamer, RedisStreamer, RedisWorker
import torch

multiprocessing.set_start_method("spawn", force=True)
BATCH_SIZE = 8

if torch.cuda.is_available():
    device = "gpu"
else:
    device = "cpu"  # in case ci environment do not have gpu

input_batch = []
vision_model = None
managed_model = None
single_output = None
batch_output = None


class ManagedVisionModel(ManagedModel):

    def init_model(self):
        self.model = VisionModel(device=device)

    def predict(self, batch):
        return self.model.batch_prediction(batch)


def setup_module(module):
    global input_batch, vision_model, managed_model, single_output, batch_output

    with open(os.path.join(DIR_PATH, "cat.jpg"), 'rb') as f:
        image_bytes = f.read()
    input_batch = [image_bytes]
    vision_model = VisionModel(device=device)
    single_output = vision_model.batch_prediction(input_batch)
    batch_output = vision_model.batch_prediction(input_batch * BATCH_SIZE)

    managed_model = ManagedVisionModel()
    managed_model.init_model()


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
    streamer = Streamer(vision_model.batch_prediction, batch_size=8, worker_num=4)
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
    worker = RedisWorker(managed_model.predict, batch_size=BATCH_SIZE)
    single_predict = worker.model_predict(input_batch)
    assert single_predict == single_output

    batch_predict = worker.model_predict(input_batch * BATCH_SIZE)
    assert batch_predict == batch_output
