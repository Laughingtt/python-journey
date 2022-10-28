#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Iterable

from arch_upai.computing.spark._table import from_hdfs, from_rdd, from_hive, from_localfs
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

SC = None


def creat_sc():
    from pyspark import SparkContext, SparkConf
    conf = SparkConf()
    conf.setAppName("hello")
    # conf.setMaster('spark://master:7077')
    conf.set("spark.driver.maxResultSize", "50g")
    conf.set("spark.executor.memory", "50g")
    conf.set("spark.driver.memory", "50g")
    conf.set("spark.executor.cores", "1")
    conf.set("spark.cores.max", "15")
    conf.set("spark.rpc.message.maxSize	", "2048")
    conf.set("spark.network.timeout	", "1000")
    conf.set("spark.executor.extraJavaOptions", "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
    global SC
    SC = SparkContext(conf=conf).getOrCreate()


def parallelize(data: Iterable, partition: int, include_key: bool, **kwargs):
    # noinspection PyPackageRequirements
    if SC is None:
        creat_sc()
    _iter = data if include_key else enumerate(data)
    rdd = SC.parallelize(_iter, partition)
    return from_rdd(rdd)


if __name__ == '__main__':
    d = parallelize(range(100), partition=8, include_key=False)
    print(d)
