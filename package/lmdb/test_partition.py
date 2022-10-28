# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test_partition.py
@time:2022/03/03

"""
import os

def _open_env(path, write=False):
    os.makedirs(path, exist_ok=True)
    return lmdb.open(Path(path).as_posix(), create=True, max_dbs=1, max_readers=1024, lock=write, sync=True,
                     map_size=10_737_418_240)


def _get_db_path(*args):
    return os.sep.join([Standalone.get_instance().data_dir, *args])


def _get_env(*args, write=False):
    _path = _get_db_path(*args)
    return _open_env(_path, write=write)