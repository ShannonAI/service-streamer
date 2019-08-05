# coding=utf-8
# Created by Meteorix at 2019/7/30
from gevent import monkey; monkey.patch_all()
from flask import Flask, request, jsonify

from service_streamer import Streamer
from bert_model import ManagedBertModel

app = Flask(__name__)
model = None
streamer = None


@app.route("/naive", methods=["POST"])
def naive_predict():
    inputs = request.form.getlist("s")
    outputs = model.predict(inputs)
    return jsonify(outputs)


@app.route("/stream", methods=["POST"])
def stream_predict():
    inputs = request.form.getlist("s")
    outputs = streamer.predict(inputs)
    return jsonify(outputs)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    streamer = Streamer(ManagedBertModel, batch_size=64, max_latency=0.1, worker_num=8, cuda_devices=(0, 1, 2, 3))

    # ThreadedStreamer for comparison
    # model = ManagedBertModel(None)
    # model.init_model()
    # streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)

    from gevent.pywsgi import WSGIServer
    WSGIServer(("0.0.0.0", 5005), app).serve_forever()
