# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test.py
@time:2021/09/23

"""
import re
import json
import argparse
import base64, pickle
from arch.api.proto.fate_data_structure_pb2 import RawEntry, RawMap, Dict
from federatedml.secureprotol.iterative_affine import IterativeAffineCipherKey, IterativeAffineCiphertext

_source = "_source"
layers = "layers"
http2 = "http2"
protobuf = "protobuf"
protobuf_field_name = "protobuf.field.name"
b64value = "b64value"
b64value_packet = "pbf.com.webank.ai.eggroll.api.networking.proxy.Packet.b64value"


class WireUnpack(object):
    def __init__(self, file_path, a_array, n_array, key_mult=2 ** 100):
        self.a_array = a_array
        self.n_array = n_array
        self.key_mult = key_mult
        self.file_path = file_path
        self.dic = {}
        self.encipher = None

    def set_encipher(self):
        self.encipher = IterativeAffineCipherKey(self.a_array, self.n_array, self.key_mult)

    def write_decode_to_file(self):
        save_file_name = self.file_path.replace(".json", "_out.json")
        with open(save_file_name, "w", encoding="utf-8") as fp:
            json.dump(self.dic, fp)

    def read_wire_to_json(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            package_file = json.load(f)
        return package_file

    def decrypt_value(self, enc_value):
        enc_value = IterativeAffineCiphertext(enc_value)
        return self.encipher.decrypt(enc_value)

    @staticmethod
    def regular_decode(b64_str=None):
        v = base64.b64decode(b64_str)
        res = pickle.loads(v)
        return res

    @staticmethod
    def dtable_decode(b64_str=None):
        v = base64.b64decode(b64_str)
        raw_map = RawMap()
        raw_map.ParseFromString(v)

        raw_map = raw_map.ListFields()[0]

        raw_map = raw_map[1]

        cipher_list = []
        for raw_entry in raw_map:
            grad_and_hess = pickle.loads(raw_entry.ListFields()[1][1])
            cipher_list.append([i.cipher for i in grad_and_hess])

        return cipher_list

    def unfold_obj(self, transfer_name, decode_res):
        if re.search("tree_node_queue", transfer_name):
            return [i.__dict__ for i in decode_res]
        elif re.search("encrypted_splitinfo_host", transfer_name):
            data = getattr(decode_res, "data", None)
            if data is not None:
                return [[i[0].cipher, i[1].cipher] for i in data[0]]
        elif re.search("federated_best_splitinfo_host", transfer_name):
            return {"plaintext": [self.decrypt_value(grad[1].cipher) for grad in decode_res],
                    "ciphertext": [i[1].cipher for i in decode_res]}
        elif re.search("final_splitinfo_host", transfer_name):
            return [i.__dict__ for i in decode_res]
        elif re.search("tree-fit", transfer_name):
            return [i.__dict__ for i in decode_res]
        elif re.search("encrypted_grad_and_hess", transfer_name):
            return {"plaintext": [[self.decrypt_value(grad) for grad in row] for row in decode_res],
                    "ciphertext": decode_res}
        else:
            return decode_res

    def find_b64_value(self, package_file):
        for package in package_file:
            protobuf_ = package[_source][layers][http2][protobuf]
            for k_field, v_field in protobuf_.items():
                if isinstance(v_field, dict):
                    str_v_field = str(v_field)
                    is_transfer_name = re.search(r'HeteroDecisionTreeTransferVariable', str_v_field)
                    is_b64value = re.search(r'Packet.b64value', str_v_field)
                    if is_transfer_name and is_b64value:
                        transfer_name_ = re.findall(r'HeteroDecisionTreeTransferVariable.(.*?)\'', str_v_field)
                        b64value_ = re.findall(r'Packet.b64value\': \'(.*?)\'', str_v_field)

                        if len(transfer_name_) > 0:
                            transfer_name = transfer_name_[-1]

                        if len(b64value_) > 0:
                            b64value = b64value_[0]

                        try:
                            decode_res = WireUnpack.regular_decode(b64value)
                            # print(res)
                        except Exception as e:
                            # print(e)
                            try:
                                decode_res = WireUnpack.dtable_decode(b64value)
                            except Exception as f:
                                # print(f)
                                break

                        decode_res = self.unfold_obj(transfer_name, decode_res)
                        self.dic[transfer_name] = {"b64value": str(b64value), "decode_res": str(decode_res)}

    def main(self):
        self.set_encipher()
        package_file = self.read_wire_to_json()
        self.find_b64_value(package_file)
        self.write_decode_to_file()


if __name__ == '__main__':
    a_array = [23898832694949915555977724542417, 1842944143782409709263250580425842360667,
               1239877844206901588711394744094,
               407765054333989079433666286771, 4079888995325050012208630399]
    n_array = [7698488986646558839749901574559001368, 237438699281696453617606386015644776659339616626,
               4837442992430157276464174475831393109474341357511641970807,
               24259842636915000673858889713575850063905160060881014706619827893182,
               1086904606352438147176987394937069563615092342992591793288306677473577713265]
    key_mult = 1267650600228229401496703205376

    path = "/Users/tian/Projects/udaiml/udai_test/wire/0917xgb-3.json"

    parser = argparse.ArgumentParser(description="Parsing caught")
    parser.add_argument("-a", "-a_array", required=True, type=str, help="a_array")
    parser.add_argument("-n", "-n_array", required=True, type=str, help="n_array")
    parser.add_argument("-k", "-key_mult", required=False, type=int, help="key_mult")
    parser.add_argument("-p", "-path", required=True, type=str, help="path")
    args = parser.parse_args()

    a_array_ = [int(_a) for _a in args.a.split(",")]
    n_array_ = [int(_n) for _n in args.a.split(",")]
    d = WireUnpack(file_path=args.p, a_array=a_array_, n_array=n_array_, key_mult=args.k if args.k else 2 ** 100)
    # d = WireUnpack(file_path=path, a_array=a_array, n_array=n_array)
    d.main()

"""
用法:
python wire.py -a "23898832694949915555977724542417,1842944143782409709263250580425842360667,1239877844206901588711394744094,407765054333989079433666286771,4079888995325050012208630399" -n "7698488986646558839749901574559001368,237438699281696453617606386015644776659339616626,4837442992430157276464174475831393109474341357511641970807,24259842636915000673858889713575850063905160060881014706619827893182,1086904606352438147176987394937069563615092342992591793288306677473577713265" -p "/data/0917xgb-3.json" -k 1267650600228229401496703205376
"""
