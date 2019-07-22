Service Streamer for deep learning web service

# QuickStart
通常深度学习的inference按batch输入会比较快
```python
outputs = model.predict(batch_inputs)
```
但是当我们搭起web service部署模型的时候，每个request是分散到来的，占不满model的batch_size。
这样无法充分利用gpu的并行性能，导致web service的QPS也上不去。

**ServiceStreamer**是一个中间件，将request排队成一个完整的batch，在送进gpu。
牺牲一定的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。

```python
from service_streamer import ThreadedStreamer
# 参数：predict函数、max_batch、max_latency
streamer = ThreadedStreamer(model.predict, 64, 0.1)
# 用streamer.predict(batch)替代model.predict(batch)
streamer.predict(batch)
```
然后你的web server需要开启多线程（或协程）即可。

短短几行代码，理论上可以实现```batch_size/batch_per_request```倍加速。 

# 分布式GPU worker

上面的例子是在web server进程中，开启子线程调用GPU，用线程间队列进行通信和排队。

实际项目中web server的性能(QPS)远高于GPU模型的性能，所以我们支持一个web server搭配多个GPU worker。

```python
from service_streamer import Streamer
# 参数：predict函数、max_batch、max_latency
streamer = Streamer(model.predict, 64, 0.1)
# 用streamer.predict(batch)替代model.predict(batch)
outputs = streamer.predict(batch)
```
``Streamer``默认采用redis进行进程间通信和排队，将大量的请求分配到多个GPU worker中处理。
再将模型batch predict的结果传回到对应的web server，并且返回到对应的http response。

```python
from service_streamer import StreamWorker, GpuWorkerManager

class GpuWorkers(GpuWorkerManager):

    @staticmethod
    def gpu_worker(index, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index % gpu_num)
        corrector = StreamWorker(model.predict, 64, max_latency=0.1)
        corrector.run()

GpuWorkers().run_workers(worker_num=8, gpu_num=4)
```
我们还提供了简单的GPU worker管理脚本，如上定义，即可启动8个GPU worker，平均分散在4个GPU卡上。

# 分布式web server

有时候，你的web server中需要进行一些cpu密集型计算，比如图像、文本预处理，再分配到gpu worker进入模型。
这时候web server的cpu资源往往会成为性能瓶颈，于是我们也提供了多web server搭配（单个或多个）gpu worker的模式。

当你的web server都在同一台服务器时，你甚至不需要改动``streamer``的代码。
只需跟任意python web server的部署一样，用``gunicorn``或``uwsgi``实现向代理和负载均衡。

当你的web server/gpu worker不在同一台服务器时，改动也很简单：指定所有web server和gpu worker公用的唯一的redis地址

```python
# todo
streamer = Streamer(model.predict, 64, 0.1, redis_broker="172.22.22.22:3217")
```

这样每个请求会负载均衡到每个web server中进行cpu预处理，然后均匀的分布到gpu worker中进行模型predict。

# 底层Future API使用
如果你使用过任意concurrent库，应该对`future`不陌生。
当你的使用场景不是web service，又想利用``service_streamer``进行排队或者分布式GPU计算，可以直接使用Future API。
```
from ifluent_english.service_streamer import Streamer
streamer = ThreadedStreamer(model.predict, 64, 0.1)

xs = []
for i in range(200):
    future = streamer.submit([["How", "are", "you", "?"], ["Fine", "."], ["Thank", "you", "."]])
    xs.append(future)

# 先拿到所有future对象，再等待异步返回
for future in xs:
    outputs = future.get()
    print(x)
```