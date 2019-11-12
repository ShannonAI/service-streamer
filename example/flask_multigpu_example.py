# coding=utf-8
# Created by Meteorix at 2019/7/30
from flask import Flask, request, jsonify
from gevent import monkey; monkey.patch_all()
from gevent.pywsgi import WSGIServer

from bert_model import ManagedBertModel, TextInfillingModel as Model
from service_streamer import Streamer

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
    streamer = Streamer(ManagedBertModel, batch_size=64, max_latency=0.1, worker_num=4, cuda_devices=(0, 1, 2, 3))
    model = Model()
    WSGIServer(("0.0.0.0", 5005), app).serve_forever()
