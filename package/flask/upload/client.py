# !/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import requests

basedir = os.path.abspath(os.path.dirname(__file__))


def upload(file_path):
    url = "http://127.0.0.1:8818/api/upload"
    file_rb = {'myfile': open(file_path, 'rb')}
    r = requests.post(url=url, files=file_rb)
    print("upload success !")


def download(filename):
    url = "http://127.0.0.1:8818/download/{}".format(filename)
    r = requests.get(url=url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
        print("download success !")


if __name__ == '__main__':
    upload(file_path='/Users/tian/Projects/my_learning/python/flask_test/123.csv')
    download("123.csv")
