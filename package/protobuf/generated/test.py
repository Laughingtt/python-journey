# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file:test.py
@time:2022/01/11

"""
import phone_pb2, people_pb2

information = people_pb2.Information()
information.id = 1
information.address = "shanghai"
information.bid = 3.1415926232323
information.is_shortsightedness = True
information.hobby.extend(["swimming", "coding", "sleep", "movie"])
information.friends.update({"bobcat": 16, "ocelot": 17})

information.people.CopyFrom(people_pb2.People(name="bob", age=18))
information.phone.CopyFrom(phone_pb2.Phone(name="huawei", number=6800))
print(information)

information_serialize = information.SerializeToString()
print("serialize:", information_serialize)
print("serialize byte:", information.ByteSize())

information_parse = people_pb2.Information()
information_parse.ParseFromString(information_serialize)
print("parse:", information_parse)
