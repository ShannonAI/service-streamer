<h1 align="center">Service Streamer</h1>

<p align="center">加速你的深度学习web服务</p>

<p align="center">
  <a href="#这是什么">这是什么</a> •
  <a href="#功能特色">功能特色</a> •
  <a href="#安装步骤">安装步骤</a> •
  <a href="#五分钟搭建bert服务">五分钟搭建BERT服务</a> •
  <a href="#api介绍">API介绍</a> •
  <a href="#基准测试">基准测试</a> •
  <a href="#常见问题">常见问题</a> •
</p>


<h6 align="center">Made by ShannonAI • :globe_with_meridians: <a href="http://www.shannonai.com/">http://www.shannonai.com/</a></h6>


<h2 align="center">这是什么</h2>

深度学习模型在训练和测试时，通常使用小批量(mini-batch)的方式将样本组装在一起，这样能充分利用GPU的并行计算特性，加快运算速度。
但在将使用了深度学习模型的服务部署上线的时候，由于用户请求通常是离散和单次的，若采取传统的循环服务器或多线程服务器， 
会造成GPU计算资源浪费，用户等待时间线性增加。更严重的是在大量并发请求时，会造成CUDA out-of-memory error，导致服务宕机。

ServiceStreamer是一个中间件，将服务请求排队组成一个完整的batch，再送进GPU运算。牺牲最小的时延（默认最大0.1s），提升整体性能，极大提高GPU利用率。

<h2 align="center">功能特色</h2>

- :hatching_chick: **简单易用**: 只需添加两三行代码即可让模型提速上十倍。
- :zap: **处理速度快**: 低延迟，专门针对速度做了优化，见 [基准测试](#基准测试)。
- :octopus: **可扩展性好**: 可轻松扩展到多GPU场景，处理大量请求，见 [分布式](#分布式gpu-worker)。
- :crossed_swords: **适用性强**: 中间件，适用于所有深度学习框架和web框架。 

<h2 align="center">安装步骤</h2>

可通过`pip`安装，要求**Python >= 3.5** :
```bash
pip install service_streamer 
```

<h2 align="center">五分钟搭建BERT服务</h2>

在本节中，我们使用一个完整的自然语言处理任务来展示，如何在五分钟搭建起**每秒处理1400个句子**的BERT服务。

``完型填空(Text Infilling)``是自然语言处理中的一个常见任务：给定一个随机挖掉几个词的句子，模型通过给定的上下文来预测出那些被挖掉的单词。

``BERT``是一个近年来广受关注的预训练语言模型。其预训练任务之一——遮蔽语言模型与完型填空任务极为相似，因此在大规模无监督语料上做过预训练的BERT，非常适合完型填空任务。

1. 首先我们定义一个完型填空模型[bert_model.py](./example/bert_model.py)，其`predict`方法接受批量的句子，并给出每个句子中`[MASK]`位置的预测结果。

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

    注意初次使用pytorch_transformers运行时需要下载BERT模型，请稍等片刻。

2. 然后使用[Flask](https://github.com/pallets/flask)将模型封装成web服务[flask_example.py](./example/flask_example.py)

    ```python
    model = TextInfillingModel()
    @app.route("/naive", methods=["POST"])
    def naive_predict():
        inputs = request.form.getlist("s")
        outputs = model.predict(inputs)
        return jsonify(outputs)
     
    app.run(port=5005)
    ```
    
    运行[flask_example.py](./example/flask_example.py)，即可得到一个朴素的web服务器
    
    ```bash
    curl -X POST http://localhost:5005/naive -d 's=Happy birthday to [MASK].' 
    ["you"]
    ```
    
    这时候你的web服务每秒钟只能完成12句请求，见[基准测试](#基准测试)

3. 下面我们通过`service_streamer`封装你的模型函数，三行代码使BERT服务的预测速度达到每秒200+句(16倍QPS)。

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
    
    同样运行[flask_example.py](./example/flask_example.py)，用[wrk](https://github.com/wg/wrk)测试一下性能
    ```bash
    wrk -t 2 -c 128 -d 20s --timeout=10s -s benchmark.lua http://127.0.0.1:5005/stream
    ...
    Requests/sec:    200.31
    ```

4. 最后，我们利用``Streamer``封装模型，启动多个GPU worker，充分利用多卡性能实现每秒1000+句(80倍QPS)

    ```python
    from service_streamer import ManagedModel, Streamer

    class ManagedBertModel(ManagedModel):

        def init_model(self):
            self.model = TextInfillingModel()

        def predict(self, batch):
            return self.model.predict(batch)

    streamer = Streamer(ManagedBertModel, batch_size=64, max_latency=0.1, worker_num=8, cuda_devices=(0, 1, 2, 3))
    app.run(port=5005, debug=False)
    ```
    
    运行[flask_multigpu_example.py](./example/flask_multigpu_example.py)这样即可启动8个gpu worker，平均分配在4张卡上


<h2 align="center">API介绍</h2>

#### 快速入门

通常深度学习的inference按batch输入会比较快

```python
outputs = model.predict(batch_inputs)
```

用**service_streamer**中间件封装```predict```函数，将request排队成一个完整的batch，再送进GPU。
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

#### 分布式GPU worker

上面的例子是在web server进程中，开启子线程作为GPU worker进行batch predict，用线程间队列进行通信和排队。

实际项目中web server的性能(QPS)远高于GPU模型的性能，所以我们支持一个web server搭配多个GPU worker进程。

```python
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
from service_streamer import ManagedModel

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

### Future API

如果你使用过任意concurrent库，应该对`future`不陌生。
当你的使用场景不是web service，又想利用``service_streamer``进行排队或者分布式GPU计算，可以直接使用Future API。

```python
from service_streamer import ThreadedStreamer
streamer = ThreadedStreamer(model.predict, 64, 0.1)

xs = []
for i in range(200):
    future = streamer.submit(["Happy birthday to [MASK]", "Today is my lucky [MASK]"])
    xs.append(future)

# 先拿到所有future对象，再等待异步返回
for future in xs:
    outputs = future.result()
    print(outputs)
```

<h2 align="center">基准测试</h2>

### 如何做基准测试

我们使用 [wrk](https://github.com/wg/wrk) 来做基准测试。

所有测试代码和脚本在 [example](./example)可以找到。

### 环境

*   gpu: Titan Xp
*   cuda: 9.0
*   pytorch: 1.1   

### 单个GPU进程

```bash
# start flask threaded server
python example/flask_example.py

# benchmark naive api without service_streamer
wrk -t 4 -c 128 -d 20s --timeout=10s -s benchmark.lua http://127.0.0.1:5005/naive
# benchmark stream api with service_streamer
wrk -t 4 -c 128 -d 20s --timeout=10s -s benchmark.lua http://127.0.0.1:5005/stream
```

| |Naive|ThreaedStreamer|Streamer|RedisStreamer
|-|-|-|-|-|
| qps | 12.78 | 207.59 | 321.70 | 372.45 |
| latency  | 8440ms | 603.35ms | 392.66ms | 340.74ms |

### 多个GPU进程

这里对比单web server进程的情况下，多gpu worker的性能，验证通信和负载均衡机制的性能损耗。
Flask多线程server已经成为性能瓶颈，故采用gevent server，代码参考[flask_multigpu_example.py](example/flask_multigpu_example.py)

```bash
wrk -t 8 -c 512 -d 20s --timeout=10s -s benchmark.lua http://127.0.0.1:5005/stream
```

| gpu_worker_num | Naive | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|11.62|211.02|362.69|365.80|
|2|N/A|N/A|488.40|609.63|
|4|N/A|N/A|494.20|1034.57|

*   ```ThreadedStreamer```由于Python GIL的限制，多worker并没有意义，仅测单gpu worker数据进行对比。
*   ```Streamer```大于2个gpu worker时，性能提升并不是线性。这是由于flask的性能问题，server进程的cpu利用率达到100，此时瓶颈是cpu而不是gpu。

### 利用Future API使用多个GPU进程

为了规避web server的性能瓶颈，我们使用[底层Future Api](#future-api)本地测试多gpu worker的benchmark，
代码参考[future_example.py](example/future_example.py)

| gpu_worker_num | Batched | ThreadedStreamer |Streamer|RedisStreamer
|-|-|-|-|-|
|1|422.883|401.01|399.26|384.79|
|2|N/A|N/A|742.16|714.781|
|4|N/A|N/A|1400.12|1356.47|

可以看出``service_streamer``的性能跟gpu worker数量几乎成线性关系，其中进程间通信的效率略高于redis通信。

<h2 align="center">常见问题</h2>

**Q:** 使用[allennlp](https://github.com/allenai/allennlp)训练得到的模型，在推理阶段，[Streamer](./service_streamer/service_streamer.py)中设置``worker_num=4``，为什么16核cpu全部跑满，且模型计算速度反而不如``worker_num=1``？

**A:** 在多进程的模型推理计算时，如果模型依赖numpy进行数据处理，且numpy默认使用了多线程，则有可能造成cpu负载过大，使得多核计算速度反而不如单核。该类问题在使用allennlp、spacy等第三方库时可能出现，可以通过设置``numpy threads``环境变量解决。
   ```python
   import os
   os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1 
   os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1 
   os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
   import numpy
   ```
   注意要将``os``环境变量的设置放在``import numpy``之前。

**Q:** 使用RedisStreamer时，在共用同一个redis broker的情况下，如果有不止一个模型，各种待处理的batch可能会有个不同的结构，从而造成冲突怎么办？

**A:** 指定prefix参数，此时会使用redis的不同频道，从而避免冲突

启动worker的方法:  
      
```python
from service_streamer import run_redis_workers_forever
from bert_model import ManagedBertModel

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_redis_workers_forever(ManagedBertModel, 64, prefix='channel_1')
    run_redis_workers_forever(ManagedBertModel, 64, prefix='channel_2')
```

接下来在另一个文件中定义streamer并得到模型结果:  
    
```python
from service_streamer import RedisStreamer

streamer_1 = RedisStreaemr(prefix='channel_1')
streamer_2 = RedisStreaemr(prefix='channel_2')

# predict
output_1 = streamer_1.predict(batch)
output_2 = streamer_2.predict(batch)
```
