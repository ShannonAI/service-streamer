<h1 align="center">Service Streamer</h1>

<p align="center">Service Streamer for deep learning web service.</p>

<p align="center">
  <a href="#what-is-it">What is it</a> •
  <a href="#highlights">Highlights</a> •
  <a href="#install">Install</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#api">API</a> •
  <a href="#benchmark">Benchmark</a> •
  
</p>


<h6 align="center">Made by ShannonAI • :globe_with_meridians: <a href="http://www.shannonai.com/">http://www.shannonai.com/</a></h6>


<h2 align="center">What is it</h2>

深度学习模型在训练和测试时，通常使用小批量(mini-batch)的方式将样本组装在一起，这样能充分利用GPU的并行计算特性，加快运算速度。
但在将使用了深度学习模型的服务部署上线的时候，由于用户请求通常是离散和单次的，若采取传统的同步阻塞式的消息通信机制，
在短时间内有大量请求时，会造成计算资源闲置，用户等待时间变长。

ServiceStreamer是一个中间件，将request排队成一个完整的batch，在送进gpu。 牺牲一定的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。

<h2 align="center">Highlights</h2>

- :hatching_chick: **简单易用**: 添加两三行代码即可跑起来。
- :zap: **处理速度快**: 低延迟，专门针对速度做了优化。见 [benchmark](#zap-benchmark).
- :octopus: **可扩展性好**: 可轻松扩展到多GPU，大量请求。见 [分布式](#分布式).
- :gem: **可靠性强**: 在大量数据集和模型上测试没有发现错误和异常。

<h2 align="center">Install</h2>

可通过`pip`安装，要求**Python >= 3.5** :
```bash
pip install service_streamer 
```

<h2 align="center">Getting Started</h2>
通常深度学习的inference按batch输入会比较快

```python
outputs = model.predict(batch_inputs)
```

但是当我们搭起web service部署模型的时候，每个request是分散到来的，占不满model的batch_size。
这样无法充分利用GPU的并行性能，导致web service的QPS也上不去。

**ServiceStreamer**是一个中间件，将request排队成一个完整的batch，在送进GPU。
牺牲一定的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。

```python
from service_streamer import ThreadedStreamer

# 用Streamer封装batch_predict函数
streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)

# 用streamer.predict替代model.predict
outpus = streamer.predict(batch_inputs)
```

然后你的web server需要开启多线程（或协程）即可。

短短几行代码，理论上可以实现```batch_size/batch_per_request```倍加速。 

<h2 align="center">分布式</h2>

#### 分布式GPU worker

上面的例子是在web server进程中，开启子线程作为GPU worker进行batch predict，用线程间队列进行通信和排队。

实际项目中web server的性能(QPS)远高于GPU模型的性能，所以我们支持一个web server搭配多个GPU worker进程。

```python
from service_streamer import Streamer

streamer = Streamer(model.predict, 64, 0.1)
outputs = streamer.predict(batch)
```
``Streamer``默认采用redis进行进程间通信和排队，将大量的请求分配到多个GPU worker中处理。
再将模型batch predict的结果传回到对应的web server，并且返回到对应的http response。

```python
from service_streamer import RedisWorker, GpuWorkerManager

class GpuWorkers(GpuWorkerManager):

    @staticmethod
    def gpu_worker(index, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index % gpu_num)
        streamer = RedisWorker(model.predict, 64, max_latency=0.1)
        streamer.run_forever()

if __name__ == "__main__":
    GpuWorkers().run_workers(worker_num=8, gpu_num=4)
```
我们还提供了简单的GPU worker管理脚本，如上定义，即可启动8个GPU worker，平均分散在4个GPU卡上。

#### 分布式web server

有时候，你的web server中需要进行一些cpu密集型计算，比如图像、文本预处理，再分配到gpu worker进入模型。
这时候web server的cpu资源往往会成为性能瓶颈，于是我们也提供了多web server搭配（单个或多个）gpu worker的模式。

当你的web server都在同一台服务器时，你甚至不需要改动``streamer``的代码。
只需跟任意python web server的部署一样，用``gunicorn``或``uwsgi``实现反向代理和负载均衡。

当你的web server/gpu worker不在同一台服务器时，改动也很简单：指定所有web server和gpu worker公用的唯一的redis地址

```python
# todo
streamer = Streamer(model.predict, 64, 0.1, redis_broker="172.22.22.22:6379")
```

这样每个请求会负载均衡到每个web server中进行cpu预处理，然后均匀的分布到gpu worker中进行模型predict。

### 底层Future API使用
如果你使用过任意concurrent库，应该对`future`不陌生。
当你的使用场景不是web service，又想利用``service_streamer``进行排队或者分布式GPU计算，可以直接使用Future API。
```python
from service_streamer import ThreadedStreamer as Streamer
streamer = Streamer(model.predict, 64, 0.1)

xs = []
for i in range(200):
    future = streamer.submit([["How", "are", "you", "?"], ["Fine", "."], ["Thank", "you", "."]])
    xs.append(future)

# 先拿到所有future对象，再等待异步返回
for future in xs:
    outputs = future.result()
    print(outputs)
```

<h2 align="center">Benchmark</h2>

### how to benchmark
We use [wrk](https://github.com/wg/wrk) to do benchmark
```shell
# start flask threaded server
python example/flask_example.py

# benchmark naive api without service_streamer
./wrk -t 4 -c 64 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/naive
# benchmark stream api with service_streamer
./wrk -t 4 -c 64 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/stream
```

### single gpu worker

| |Naive|ThreaedStreamer|Streamer|RedisStreamer
|-|-|-|-|-|
| qps | 12.78 | 151.96 |  | |
| latency  | 186.92ms | 112.68ms |  |  | |
