# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:client.py
@time:2022/07/08

"""
import pickle
from network.client import GrpcChannelClient


class Client(object):
    def __init__(self):
        self._channel = GrpcChannelClient(addr_and_port="localhost:50033")

    def remote(self, data):
        if not isinstance(data, bytes):
            pickle_data = pickle.dumps(data)
        else:
            pickle_data = data
        self._channel.send(data=pickle_data)

    def get(self):
        pickle_data = self._channel.recv()
        return pickle.loads(pickle_data)


class Sender(object):
    def __init__(self):
        self.client = Client()

    def send(self, msg_list):
        self.client.remote(msg_list)

    @staticmethod
    def serialize_msg(msg_list):
        return [pickle.dumps(msg) for msg in msg_list]


if __name__ == '__main__':
    msg_list = [1, 2, 3, 4, 5, 6]
    for msg in msg_list:
        print("msg :", msg)
    S = Sender()
    S.send(msg_list=msg_list)
