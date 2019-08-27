# coding=utf-8
# Created by Meteorix at 2019/7/31
from gevent import monkey; monkey.patch_all()


def post_fork(server, worker):
    from service_streamer import RedisStreamer
    import flask_example
    flask_example.streamer = RedisStreamer()


bind = '0.0.0.0:5005'
workers = 4
worker_class = 'gunicorn.workers.ggevent.GeventWorker'
proc_name = "redis_streamer"
