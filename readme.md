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
- :zap: **处理速度快**: 低延迟，专门针对速度做了优化。见 [benchmark](#Benchmark).
- :octopus: **可扩展性好**: 可轻松扩展到多GPU，大量请求。见 [分布式](#分布式).
- :gem: **可靠性强**: 在大量数据集和模型上测试没有发现错误和异常。

<h2 align="center">Install</h2>

可通过`pip`安装，要求**Python >= 3.5** :
```bash
pip install service_streamer 
```

<h2 align="center">Example</h2>
我们提供了一个完整的[example](./example)，利用PyTorch实现的Bert预测下一个词的服务。
并且针对这个example做了性能[benchmark](#Benchmark)。

<h2 align="center">Getting Started</h2>
通常深度学习的inference按batch输入会比较快

```python
outputs = model.predict(batch_inputs)
```

用**ServiceStreamer**中间件封装```predict```函数，将request排队成一个完整的batch，再送进GPU。
牺牲一定的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。

```python
from service_streamer import ThreadedStreamer

# 用Streamer封装batch_predict函数
streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)

# 用streamer.predict替代model.predict
outpus = streamer.predict(batch_inputs)
```

然后你的web server需要开启多线程（或协程）即可。

短短几行代码，通常可以实现数十(```batch_size/batch_per_request```)倍的加速。 

<h2 align="center">分布式</h2>

#### 分布式GPU worker

上面的例子是在web server进程中，开启子线程作为GPU worker进行batch predict，用线程间队列进行通信和排队。

实际项目中web server的性能(QPS)远高于GPU模型的性能，所以我们支持一个web server搭配多个GPU worker进程。

```python
import multiprocessing; multiprocessing.set_start_method("spawn", force=True)
from service_streamer import Streamer

# spawn出4个gpu worker进程
streamer = Streamer(model.predict, 64, 0.1, worker_num=4)
outputs = streamer.predict(batch)
```
``Streamer``默认采用``spawn``子进程运行gpu worker，利用进程间队列进行通信和排队，将大量的请求分配到多个worker中处理。
再将模型batch predict的结果传回到对应的web server，并且返回到对应的http response。

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

上面这种方式定义简单，但是主进程初始化模型，多占了一份显存，并且模型只能运行在同一块GPU上。
所以我们提供了```ManagedModel```类，方便模型lazy初始化和迁移，以支持多GPU卡。

```python
class ManagedBertModel(ManagedModel):

    def init_model(self):
        self.model = Model()

    def predict(self, batch):
        return self.model.predict(batch)


# spawn出4个gpu worker进程，平均分散在0/1/2/3号GPU上
streamer = Streamer(ManagedBertModel, 64, 0.1, worker_num=4, cuda_devices=(0, 1, 2, 3))
outputs = streamer.predict(batch)
```

#### 分布式web server

有时候，你的web server中需要进行一些cpu密集型计算，比如图像、文本预处理，再分配到gpu worker进入模型。
cpu资源往往会成为性能瓶颈，于是我们也提供了多web server搭配（单个或多个）gpu worker的模式。

使用```RedisStreamer```指定所有web server和gpu worker公用的唯一的redis地址

```python
# 默认参数可以省略，使用localhost:6379
streamer = RedisStreamer(redis_broker="172.22.22.22:6379")
```

然后跟任意python web server的部署一样，用``gunicorn``或``uwsgi``实现反向代理和负载均衡。

```bash
cd example
gunicorn -c redis_streamer_gunicorn.py flask_example:app
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

All the code and bench scripts are in [example](./example).

### environment

*   cpu: 
*   gpu: Titan Xp
*   cuda: 9.0
*   pytorch: 1.1   

### single gpu worker

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

### multiple gpu workers

这里对比单web server进程的情况下，多gpu worker的性能，验证通信和负载均衡机制的性能损耗。
Flask多线程server已经成为性能瓶颈，故采用gevent server，代码参考[flask_multigpu_example.py](example/flask_multigpu_example.py)

```bash
./wrk -t 8 -c 512 -d 20s --timeout=10s -s scripts/streamer.lua http://127.0.0.1:5005/stream
```

| gpu_worker_num | Naive | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|11.62|211.02|362.69|365.80|
|2|N/A|N/A|488.40|609.63|
|4|N/A|N/A|494.20|1034.57|

*   ```ThreadedStreamer```由于Python GIL的限制，多worker并没有意义，仅测单gpu worker数据进行对比。
*   ```Streamer```大于2个gpu worker时，性能提升并不是线性。这是由于flask的性能问题，server进程的cpu利用率达到100，此时瓶颈是cpu而不是gpu。

### multiple gpu workers future api

为了规避web server的性能瓶颈，我们使用[底层Future Api](#底层Future API使用)本地测试多gpu worker的benchmark，
代码参考[future_example.py](example/future_example.py)

| gpu_worker_num | Batched | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|422.883|401.01|399.26|384.79|
|2|N/A|N/A|742.16|714.781|
|4|N/A|N/A|1400.12|1356.47|

可以看出``service_streamer``的性能跟gpu worker数量几乎成线性关系，其中进程间通信的效率略高于redis通信。
