# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:server.py
@time:2022/07/08

"""
import pickle
from network.server import GrpcChannelServer


class Server(object):
    def __init__(self):
        self._channel = GrpcChannelServer(addr_and_port="localhost:50033")

    def kill(self):
        self._channel.stop()

    def remote(self, data):
        if not isinstance(data, bytes):
            pickle_data = pickle.dumps(data)
        else:
            pickle_data = data
        self._channel.send(data=pickle_data)

    def get(self):
        pickle_data = self._channel.recv()
        return pickle.loads(pickle_data)


class Receiver(object):

    def __init__(self):
        self.server = Server()

    def receive(self):
        msg = self.server.get()
        return msg

    @staticmethod
    def deserialize_msg(msg):
        return pickle.loads(msg)


if __name__ == '__main__':
    R = Receiver()
    msg = R.receive()
    print(" msg:", msg)
