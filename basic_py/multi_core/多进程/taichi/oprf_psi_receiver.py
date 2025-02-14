import time
import syspath
import numpy as np
from Crypto.Cipher import AES
from bitarray import bitarray
from numpy import ceil, log2, logical_xor, random
from protocol.oprf.oprf_base import OprfBase
from protocol.oprf.transfer.client import Client
from protocol.oprf.core.statics_method import int_list_to_byte
from bitarray import bitarray

import taichi as ti

ti.init()


@ti.kernel
def insert_position(D_matrix: ti.types.ndarray(), v_list: ti.types.ndarray()):
    print(D_matrix.shape)
    print(v_list.shape)
    param_w = D_matrix.shape[0]
    param_m = D_matrix.shape[1]
    for idx in range(param_m):
        for wx in range(param_w):
            D_matrix[wx, v_list[idx, wx]] = 0


class PsiReceiver(OprfBase):

    def __init__(self, param_lambda, param_sigma, param_m, param_w, param_len1, param_len2):
        super().__init__()
        self.param_lambda = param_lambda
        self.param_sigma = param_sigma
        self.param_m = param_m
        self.param_w = param_w
        self.param_len1 = param_len1
        self.param_len2 = param_len2
        self.param_t = int(ceil(self.param_w * log2(self.param_m) / self.param_lambda))
        self.transfer_variable = Client()

    def pre_calculation(self, dataset):
        PRF_key = np.random.randint(1, size=self.param_lambda, dtype=np.uint8)
        PRF_key_bytes = int_list_to_byte(PRF_key)
        cipher = AES.new(PRF_key_bytes, AES.MODE_CTR)
        PRF_nonce = cipher.nonce
        PRF_list = [PRF_key_bytes, PRF_nonce]

        # send PRF Key
        self.transfer_variable.remote(PRF_list)
        print("send prf key complete")

        v_list = self.evaluation_cal_v(PRF_key_bytes, PRF_nonce, dataset)

        print("cal v_list complete")

        D_matrix = self.insert_d_matrix(v_list)

        return v_list, D_matrix

    def insert_d_matrix(self, v_list):
        t0 = time.time()
        ## py
        # one_list = bitarray(np.ones(self.param_m, dtype=np.uint8).tolist())
        # D_matrix = [one_list for w in range(self.param_w)]
        # for v_item in v_list:
        #     for i in range(len(v_item)):
        #         D_matrix[i][v_item[i]] = 0
        #
        # ## cpp
        # from udaiml_lib import Utils
        # D_matrix = Utils.insert_d_matrix(self.param_m, self.param_w, v_list)
        # D_matrix = [bitarray(d) for d in D_matrix]

        # taichi
        D_matrix = np.ones((self.param_w, self.param_m), dtype=np.uint32)
        insert_position(D_matrix, v_list)
        print("insert time :{}".format(time.time() - t0))
        D_matrix = [bitarray(D_matrix[0].tolist()) for i in D_matrix]
        return D_matrix

    def base_OT_prepare_sender(self, D_matrix):
        # generate matrix A
        A_matrix = np.random.randint(2, size=(self.param_w, self.param_m), dtype=np.uint8)
        B_matrix = logical_xor(A_matrix, D_matrix).astype(np.uint8)
        result_dic = {"A_matrix": A_matrix, "B_matrix": B_matrix}
        return result_dic

    def run(self, dataset):
        # 1. call pre-calculation
        t0 = time.time()
        v_list, D_matrix = self.pre_calculation(dataset)
        print("pre_calculation time is {}".format(time.time() - t0))

        # 2b. Oblivious Transfer: run np sender
        t0 = time.time()
        random_mesgs = self.transfer_variable.ot_sender_base(self.param_w, self.param_m)
        print("ot_sender base time is {}".format(time.time() - t0))

        t0 = time.time()

        A_matrix = self.transfer_variable.ot_sender_ext(random_mesgs, D_matrix)
        print("ot_sender base time is {}".format(time.time() - t0))

        H2_x = self.transfer_variable.get()

        # 4. Calculate Hash2 of y & compare & output same result
        t0 = time.time()
        H2_y = self.evaluation_cal_Psi(v_list, A_matrix)
        print("evaluation_cal_Psi time is {}".format(time.time() - t0))

        t0 = time.time()
        intersection_hash = list(set(H2_x).intersection(H2_y))
        intersection_mesg = []
        hash_data_dict = dict(zip(H2_y, dataset))
        for hash in intersection_hash:
            intersection_mesg.append(hash_data_dict.get(hash))

        print("intersect count  is :", len(intersection_mesg), time.time() - t0,
              "\n=================================================\n")


if __name__ == "__main__":
    size = int(100000)
    loop = 1
    t1 = 0.00
    receiver = PsiReceiver(128, 40, size, 609, 256, 72)
    for j in range(loop):
        set_Y = [str(i + j) for i in range(size)]
        t0 = time.time()
        receiver.run(set_Y)
        t1 += time.time() - t0

    print("time is {}".format(t1 / loop))
    """
    insert time :0.22525787353515625
    pre_calculation time is 1.239461898803711
    ot_sender base time is 0.20873403549194336
    ot_sender base time is 0.00664210319519043
    evaluation_cal_Psi time is 0.35503506660461426
    intersect count  is : 10000 0.004094123840332031 
    =================================================
    
    time is 2.383607864379883
    """
