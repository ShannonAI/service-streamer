# coding=utf-8
# Created by Meteorix at 2019/8/16
from service_streamer import ThreadedStreamer, ManagedModel
from vision_case.model import VisionModel, DIR_PATH
import os


input_batch = []
vision_model = None
managed_model = None


def setup_module(module):
    global input_batch, vision_model, managed_model

    device = "cpu"  # in case ci environment do not have gpu
    with open(os.path.join(DIR_PATH, "cat.jpg"), 'rb') as f:
        image_bytes = f.read()
    input_batch = [image_bytes]
    vision_model = VisionModel(device=device)

    class ManagedVisionModel(ManagedModel):

        def init_model(self):
            self.model = VisionModel(device=device)

        def predict(self, batch):
            return self.model.batch_prediction(batch)

    managed_model = ManagedVisionModel()
    managed_model.init_model()


def test_threaded_streamer():
    streamer = ThreadedStreamer(vision_model.batch_prediction, batch_size=16)

    output_raw = vision_model.batch_prediction(input_batch)
    output = streamer.predict(input_batch)
    assert output_raw == output

    outputs_raw = vision_model.batch_prediction(input_batch * 55)
    outputs = streamer.predict(input_batch * 55)
    assert outputs_raw == outputs


def test_managed_model():
    output_raw = vision_model.batch_prediction(input_batch)
    output = managed_model.predict(input_batch)
    assert output_raw == output

    outputs_raw = vision_model.batch_prediction(input_batch * 16)
    outputs = managed_model.predict(input_batch * 16)
    assert outputs_raw == outputs


def test_spawned_streamer():
    # TODO
    pass


def test_redis_streamer():
    # TODO
    pass


def test_future_api():
    # TODO
    pass
