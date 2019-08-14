# Vision Example for ServiceStreamer

## tutorial

[Develop Vision Service with Flask and service streamer](https://github.com/ShannonAI/service-streamer/wiki/Develop-Vision-Service-with-Flask-and-service-streamer)

## install 
```bash
pip install torchvision pillow flask gevent service_streamer
```

## start server
```bash
python app.py

# test the api
curl -F "file=@cat.jpg" http://localhost:5005/predict
{"class_id":"n02123045","class_name":"tabby"}
```

## benchmark
```bash
./wrk -t 2 -c 128 -d 20s --timeout=20s -s file.lua http://127.0.0.1:5005/predict
```
