#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: Wei Wu 
@license: Apache Licence 
@file: bert_client.py 
@time: 2019/08/21
@contact: wu.wei@pku.edu.cn

"""
import requests
from example.bert_extracting_features import batch

url = 'http://0.0.0.0:5005/stream'
result = requests.post(url, json={'instances': batch}).json()
print(result)
