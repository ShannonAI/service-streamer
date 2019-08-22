# coding=utf-8
# Created by Meteorix at 2019/7/30

import multiprocessing as mp
from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer
from example.bert_extracting_features import BertExtractor as Model

app = Flask(__name__)
model = None
streamer = None


@app.route("/naive", methods=["POST"])
def naive_predict():
    instances = request.json.get('instances')
    outputs = model.predict(batch=instances)
    return jsonify(outputs)


@app.route("/stream", methods=["POST"])
def stream_predict():
    instances = request.json.get('instances')
    outputs = streamer.predict(batch=instances)
    return 'OK'


if __name__ == "__main__":
    # for ThreadedStreamer/Streamer
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    model = Model()
    streamer = ThreadedStreamer(model.predict, batch_size=128, max_latency=0.1)
    # streamer = Streamer(model.predict, batch_size=64, max_latency=0.1)

    app.run(port=5005, debug=False)
