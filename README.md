<h1 align="center">Service Streamer</h1>

<p align="center">Boosting your Web Services of Deep Learning Applications. <a href="./README_zh.md">中文README</a></p>


<p align="center">
  <a href="#What is Service Streamer ?">What is Service Streamer ?</a> •
  <a href="#Highlights">Highlights</a> •
  <a href="#Installation">Installation</a> •
  <a href="#Develop BERT Service in 5 Mintues">Develop BERT Service in 5 Mintues</a> •
  <a href="#API">API</a> •
  <a href="#Benchmark">Benchmark</a> •
  
</p>


<h6 align="center">Made by ShannonAI • :globe_with_meridians: <a href="http://www.shannonai.com/">http://www.shannonai.com/</a></h6>


<h2 align="center">What is Service Streamer ?</h2>

A mini-batch collects individual data samples and is usually adopted in deep learning models during training and inference. Models can utilize the parallel computation characteristic of GPUs and speed up computing. Requests from users are usually discrete when machine learning models are deployed online. There is an issue that computing processors are idle when using conventional synchronous blocking message communication mechanism. The wait time will be longer when there are requests from enormous users in a short time. 

ServiceStreamer is a middleware for web service of machine learning applications. Queue requests from users are scheduled into mini-batches. ServiceStreamer can enhance the overall performance of the system by improving the ratio of GPU utilization. 

<h2 align="center">Highlights</h2>

- :hatching_chick: **Easy to use**: Minor changes can speed up the model ten times. 
- :zap: **Fast processing speed**: Low latency for online inference of machine learning models. 
- :octopus: **Good expandability**: Easy to be applied to multi-GPU scenarios for handling enormous requests.
- :crossed_swords: **Applicability**: Used with any web frameworks and/or deep learning frameworks. 


<h2 align="center">Installation</h2>

Install ServiceStream by using `pip`，requires **Python >= 3.5** :
```bash
pip install service_streamer 
```

<h2 align="center">Develop BERT Service in 5 Mintues</h2>

We provide a step-by-step tutorial for you to bring BERT online in 5 minutes. The service processes 1400 sentences per second.  

``Text Infilling`` is a task in natural language processing: given a sentence with several words randomly removed, the model predicts those words removed through the given context. 

``BERT`` has attracted a lot of attention in these two years and it achieves new State-Of-The-Art results across many nlp tasks. BERT utilizes "Masked Language Model (MLM)" as one of the pre-training objectives. MLM models randomly mask some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based on its context. MLM has similarities with text infilling. It is natural to introduce BERT to text infilling task. 


1. First, we define a model for text filling task [bert_model.py](./example/bert_model.py). The `predict` function accepts a batch of sentences and returns predicted position results of the `[MASK]` token. 

    ```python
    class TextInfillingModel(object):
        ...


    batch = ["twinkle twinkle [MASK] star.",
             "Happy birthday to [MASK].",
             'the answer to life, the [MASK], and everything.']
    model = TextInfillingModel()
    outputs = model.predict(batch)
    print(outputs)
    # ['little', 'you', 'universe']
    ```
    **Note**: Please download pre-trained BERT model at first. 


2. Second, utilize [Flask](https://github.com/pallets/flask) to pack predicting interfaces to Web service. [flask_example.py](./example/flask_example.py)


    ```python
    model = TextInfillingModel()
    @app.route("/naive", methods=["GET", "POST"])
    def naive_predict():
        if request.method == "GET":
            inputs = request.args.getlist("s")
        else:
            inputs = request.form.getlist("s")
        outputs = model.predict(inputs)
        return jsonify(outputs)
     
    app.run(port=5005)
    ```
    
    Please run [flask_example.py](./example/flask_example.py), then you will get a vanilla Web server. 

    ```bash
    curl -X POST http://localhost:5005/naive -d 's=Happy birthday to [MASK].' 
    ["you"]
    ```

    At this time, your web server can only serve 12 requests per second. Please see [benchmark](#benchmark) for more details.


3. Third, encapsulate model functions through `service_streamer`. Three lines of code make the prediction speed of BERT service reach 200+ sentences per second (16x faster).

    ```python
    from service_streamer import ThreadedStreamer
    streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)

    @app.route("/stream", methods=["POST"])
    def stream_predict():
        inputs = request.form.getlist("s")
        outputs = streamer.predict(inputs)
        return jsonify(outputs)

    app.run(port=5005, debug=False)
    ```

    Run [flask_example.py](./example/flask_example.py) and test the performance with [wrk](https://github.com/wg/wrk). 

    ```bash
    ./wrk -t 2 -c 128 -d 20s --timeout=10s -s example/benchmark.lua http://127.0.0.1:5005/stream
    ...
    Requests/sec:    200.31
    ```

4. Finally, encapsulate models through ``Streamer`` and start service workers on multiple GPUs. ``Streamer`` further accelerates inference speed and achieves 1000+ sentences per second (80x faster). 



    ```python
    import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
    from service_streamer import ManagedModel, Streamer
    multiprocessing.set_start_method("spawn", force=True)

    class ManagedBertModel(ManagedModel):

        def init_model(self):
            self.model = TextInfillingModel()

        def predict(self, batch):
            return self.model.predict(batch)

    streamer = Streamer(ManagedBertModel, batch_size=64, max_latency=0.1, worker_num=8, cuda_devices=(0, 1, 2, 3))
    app.run(port=5005, debug=False)
    ```

    8 gpu workers can be started and evenly distributed on 4 GPUs.


<h2 align="center">API</h2>

#### Quick Start

In general, the inference speed will be faster by utilizing parallel computing.

```python
outputs = model.predict(batch_inputs)
```

**ServiceStreamer** is a middleware for web service of machine learning applications. Queue requests from users are scheduled into mini-batches and forward into GPU workers. ServiceStreamer sacrifices a certain delay (default maximum is 0.1s) and enhance the overall performance by improving the ratio of GPU utilization. 


```python
from service_streamer import ThreadedStreamer

# Encapsulate batch_predict function with Streamer

streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)

# Replace model.predict with streamer.predict

outputs = streamer.predict(batch_inputs)
```

Start web server on multi-threading (or coordination). Your server can usually achieve 10x (```batch_size/batch_per_request```) times faster by adding a few lines of code.


#### Distributed GPU worker

The performance of web server (QPS) in actual projects is much higher than that of GPU model, so we support one web server with multiple GPU worker processes.

```python
import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
from service_streamer import Streamer

# Spawn releases 4 gpu worker processes
streamer = Streamer(model.predict, 64, 0.1, worker_num=4)
outputs = streamer.predict(batch)
```


``Streamer`` uses ``spawn`` subprocesses to run gpu workers by default. ``Streamer`` uses interprocess queues to communicate and queue. It can distribute a large number of requests to multiple workers for processing.

Then the prediction results of the model are returned to the corresponding web server in batches. And results are forwarded to the corresponding http response.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.116                Driver Version: 390.116                   |
|-------------------------------+----------------------+----------------------+
...
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      7574      C   /home/liuxin/nlp/venv/bin/python            1889MiB |
|    1      7575      C   /home/liuxin/nlp/venv/bin/python            1889MiB |
|    2      7576      C   /home/liuxin/nlp/venv/bin/python            1889MiB |
|    3      7577      C   /home/liuxin/nlp/venv/bin/python            1889MiB |
+-----------------------------------------------------------------------------+

```

The above method is simple to define, but the main process initialization model takes up an extra portion of memory. And the model can only run on the same GPU.
Therefore, we have provided the ```ManagedModel``` class to facilitate model lazy initialization and migration while supporting multiple GPUs.

```python
import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
from service_streamer import ManagedModel

class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = Model()

    def predict(self, batch):
        return self.model.predict(batch)


# Spawn produces 4 gpu worker processes, which are evenly distributed on 0/1/2/3 GPU
streamer = Streamer(ManagedBertModel, 64, 0.1, worker_num=4, cuda_devices=(0, 1, 2, 3))
outputs = streamer.predict(batch)
```

#### Distributed Web Server

Some cpu-intensive calculations, such as image and text preprocessing, need to be done first in web server. The preprocessed data is then forward into GPU worker for predictions after preprocessing. CPU resources often become performance bottlenecks in practice. Therefore, we also provide the mode of multi-web servers matching (single or multiple) gpu worker.


Use ```RedisStream``` to specify a unique Redis address for all web servers and gpu workers. 


```python
# default parameters can be omitted and localhost:6379 is used.
streamer = RedisStreamer(redis_broker="172.22.22.22:6379")
```


We make use of ``gunicorn`` or ``uwsgi`` to implement reverse proxy and load balancing.

```bash
cd example
gunicorn -c redis_streamer_gunicorn.py flask_example:app
```

Each request will be load balanced to each web server for cpu preprocessing, and then evenly distributed to gpu worker for model prediction.


### Future API

You might be familiar with `future` if you have used any concurrent library. 
You can directly use the Future API when you want to use ``service_streamer`` for queueing requests or distributed GPU computing and your usage scenario is not web service. 


```python
from service_streamer import ThreadedStreamer
streamer = ThreadedStreamer(model.predict, 64, 0.1)

xs = []
for i in range(200):
    future = streamer.submit([["How", "are", "you", "?"], ["Fine", "."], ["Thank", "you", "."]])
    xs.append(future)


# Get all instances of future object and wait for asynchronous responses. 
for future in xs:
    outputs = future.result()
    print(outputs)
```

<h2 align="center">Benchmark</h2>

### Benchmark

We utilize [wrk](https://github.com/wg/wrk) to conduct benchmark test.

Test examples and scripts can be found in [example](./example).


### Environment

*   gpu: Titan Xp
*   cuda: 9.0
*   pytorch: 1.1   

### Single GPU process

```bash
# start flask threaded server
python example/flask_example.py

# benchmark naive api without service_streamer
./wrk -t 4 -c 128 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/naive
# benchmark stream api with service_streamer
./wrk -t 4 -c 128 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/stream
```

| |Naive|ThreaedStreamer|Streamer|RedisStreamer
|-|-|-|-|-|
| qps | 12.78 | 207.59 | 321.70 | 372.45 |
| latency  | 8440ms | 603.35ms | 392.66ms | 340.74ms |

### Multiple GPU processes

The performance loss of the communications and load balancing mechanism of multi-gpu workers are verified compared with a single web server process.

We adopt gevent server because multi-threaded Flask server has become a performance bottleneck. Please refer to the [flask_multigpu_example.py](example/flask_multigpu_example.py)


```bash
./wrk -t 8 -c 512 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/stream
```

| gpu_worker_num | Naive | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|11.62|211.02|362.69|365.80|
|2|N/A|N/A|488.40|609.63|
|4|N/A|N/A|494.20|1034.57|


* ``Threaded Streamer`` Due to the limitation of Python GIL, multi-worker is meaningless. We conduct comparison studies between single GPU worker. 

* ``Streamer`` Performance improvement is not linear when it is greater than 2 gpu worker.
The utilization rate of CPU reaches 100 and the bottleneck is CPU at this time. And the performance issue of flask is the obstacle.  



### Utilize Future API to start multiple GPU processes

We adopt [Future API](#future-api) to conduct multi-GPU benchmeark test locally in order to reduce the influence of performance bottleneck of web server. Please refer to code example in [future_example.py](example/future_example.py)


| gpu_worker_num | Batched | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|422.883|401.01|399.26|384.79|
|2|N/A|N/A|742.16|714.781|
|4|N/A|N/A|1400.12|1356.47|

It can be seen that the performance of ``service_streamer`` is almost linearly related to the number of gpu workers. And the efficiency of inter-process communication is slightly higher than that of redis communication.
