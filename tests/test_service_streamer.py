# coding=utf-8
# Created by Meteorix at 2019/8/16
import os
import threading

from vision_case.model import VisionDensenetModel, VisionResNetModel, DIR_PATH

from service_streamer import ThreadedStreamer, ManagedModel, Streamer, RedisStreamer, RedisWorker, \
    run_redis_workers_forever
import torch
import pytest

BATCH_SIZE = 2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # in case ci environment do not have gpu


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


class TestClass(object):

    def setup_class(self):
        with open(os.path.join(DIR_PATH, "cat.jpg"), 'rb') as f:
            image_bytes = f.read()
        self.input_batch = [image_bytes]
        self.vision_model = VisionDensenetModel(device=device)
        self.single_output = self.vision_model.batch_prediction(self.input_batch)
        self.batch_output = self.vision_model.batch_prediction(self.input_batch * BATCH_SIZE)

        with open(os.path.join(DIR_PATH, "dog.jpg"), 'rb') as f:
            image_bytes2 = f.read()
        self.input_batch2 = [image_bytes2]
        self.vision_model2 = VisionResNetModel(device=device)
        self.single_output2 = self.vision_model2.batch_prediction(self.input_batch2)
        self.batch_output2 = self.vision_model2.batch_prediction(self.input_batch2 * BATCH_SIZE)

        self.managed_model = ManagedVisionDensenetModel()
        self.managed_model.init_model()

    def test_init_redis_workers(self):
        thread = threading.Thread(target=run_redis_workers_forever, args=(
            ManagedVisionDensenetModel, 8, 0.1, 2, (0, 1, 2, 3), "localhost:6379", ''), daemon=True)
        thread1 = threading.Thread(target=run_redis_workers_forever, args=(
            ManagedVisionDensenetModel, 8, 0.1, 2, (0, 1, 2, 3), "localhost:6379", 'channel_for_densenet'), daemon=True)
        thread2 = threading.Thread(target=run_redis_workers_forever, args=(
            ManagedVisionResNetModel, 8, 0.1, 2, (0, 1, 2, 3), "localhost:6379", 'channel_for_resnet'), daemon=True)

        thread.start()
        thread1.start()
        thread2.start()

    def test_threaded_streamer(self):
        streamer = ThreadedStreamer(self.vision_model.batch_prediction, batch_size=8)
        single_predict = streamer.predict(self.input_batch)
        assert single_predict == self.single_output

        batch_predict = streamer.predict(self.input_batch * BATCH_SIZE)
        assert batch_predict == self.batch_output

        streamer.destroy_workers()

    def test_managed_model(self):
        single_predict = self.managed_model.predict(self.input_batch)
        assert single_predict == self.single_output

        batch_predict = self.managed_model.predict(self.input_batch * BATCH_SIZE)
        assert batch_predict == self.batch_output

    def test_spawned_streamer(self):
        # Spawn releases 4 gpu worker processes
        streamer = Streamer(self.vision_model.batch_prediction, batch_size=8, worker_num=4, cuda_devices=(0, 1, 2, 3))
        single_predict = streamer.predict(self.input_batch)
        assert single_predict == self.single_output

        batch_predict = streamer.predict(self.input_batch * BATCH_SIZE)
        assert batch_predict == self.batch_output

        streamer.destroy_workers()

    def test_future_api(self):
        streamer = ThreadedStreamer(self.vision_model.batch_prediction, batch_size=8)

        xs = []
        for i in range(BATCH_SIZE):
            future = streamer.submit(self.input_batch)
            xs.append(future)
        batch_predict = []
        # Get all instances of future object and wait for asynchronous responses.
        for future in xs:
            batch_predict.extend(future.result())
        assert batch_predict == self.batch_output

        streamer.destroy_workers()

    def test_redis_streamer(self):
        # Spawn releases 4 gpu worker processes
        streamer = RedisStreamer()
        single_predict = streamer.predict(self.input_batch)
        assert single_predict == self.single_output

        batch_predict = streamer.predict(self.input_batch * BATCH_SIZE)
        assert batch_predict == self.batch_output

        with pytest.raises(NotImplementedError):
            streamer.destroy_workers()

    def test_multi_channel_streamer(self):
        streamer_1 = RedisStreamer(prefix='channel_for_densenet')
        streamer_2 = RedisStreamer(prefix='channel_for_resnet')

        single_predict = streamer_1.predict(self.input_batch)
        assert single_predict == self.single_output

        batch_predict = streamer_1.predict(self.input_batch * BATCH_SIZE)
        assert batch_predict == self.batch_output

        single_predict2 = streamer_2.predict(self.input_batch2)
        assert single_predict2 == self.single_output2

        batch_predict2 = streamer_2.predict(self.input_batch2 * BATCH_SIZE)
        assert batch_predict2 == self.batch_output2
