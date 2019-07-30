# coding=utf-8
# Created by Meteorix at 2019/7/30

from bert_model import Model
from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer as Streamer


app = Flask(__name__)
model = Model()
streamer = Streamer(model.predict, batch_size=64, max_latency=0.1)


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
    app.run(port=5005, threaded=True)
