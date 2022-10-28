# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:federation_obj.py
@time:2021/12/08

"""
import pickle
from arch_upai.computing.spark._table import Table
from federatedml.util.consts import CPU_COUNT
from arch_upai.computing.spark._csession import parallelize
from arch.api.utils.splitable import segment_transfer_enabled


class TransferObj:
    def __init__(self, data: Table):
        self.__data = pickle.dumps(list(data.collect()))

    def get_data(self):
        return parallelize(pickle.loads(self.__data), include_key=True, partition=CPU_COUNT)


class TransferFile(metaclass=segment_transfer_enabled(0xeffe200)):
    def __init__(self, _transfer_obj, cls, *args, **kwargs):
        self._transfer_obj = _transfer_obj
        self._cls = cls
        if args:
            self._args = args
        if kwargs:
            self._kwargs = kwargs

    def get_data(self):
        rdd = parallelize(self._transfer_obj, include_key=True, partition=CPU_COUNT)
        self._transfer_obj = None
        return rdd


class FactoryTransferFile(metaclass=segment_transfer_enabled(0xeffe200)):

    def __init__(self, _transfer_obj):
        self._transfer_obj = _transfer_obj

    def for_remote(self):
        return TransferFile(list(self._transfer_obj.collect()), self.__class__)

    def __del__(self):
        self._transfer_obj = None

    @property
    def unboxed(self):
        return self._transfer_obj


if __name__ == '__main__':
    d = parallelize(range(100), partition=8, include_key=False)
    dd = TransferObj(d)
    print(dd.get_data().first())
