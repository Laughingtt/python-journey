import hashlib
import numpy as np

zb2map = {}


def create_md5(num):
    m2 = hashlib.md5()
    m2.update(str(num).encode("utf-8"))
    return m2.hexdigest()[:4]


lis = [i for i in range(-20, 21)]
for i in range(-20, 21):
    zb2map[create_md5(i)] = i
print(zb2map)
zb2map = {'8c5ef576c6': -20, 'b9ea889c6f': -19, '74c803ab28': -18, '7ab4babdd6': -17, '55ef51c35b': -16,
          '0f4b2d5011': -15, '66d8c0f4ab': -14, '4e1ed310e8': -13, '29fe3cef22': -12, 'fe9bea9298': -11,
          '1b0fd9efa5': -10, '252e691406': -9, 'a8d2ec85ea': -8, '74687a12d3': -7, '596a3d0448': -6, '47c1b025fa': -5,
          '0267aaf632': -4, 'b3149ecea4': -3, '5d7b9adcbe': -2, '6bb61e3b7b': -1, 'cfcd208495': 0, 'c4ca4238a0': 1,
          'c81e728d9d': 2, 'eccbc87e4b': 3, 'a87ff679a2': 4, 'e4da3b7fbb': 5, '1679091c5a': 6, '8f14e45fce': 7,
          'c9f0f895fb': 8, '45c48cce2e': 9, 'd3d9446802': 10, '6512bd43d9': 11, 'c20ad4d76f': 12, 'c51ce410c1': 13,
          'aab3238922': 14, '9bf31c7ff0': 15, 'c74d97b01e': 16, '70efdf2ec9': 17, '6f4922f455': 18, '1f0e3dad99': 19,
          '98f1370821': 20}

zb2map = {'8c5e': -20, 'b9ea': -19, '74c8': -18, '7ab4': -17, '55ef': -16, '0f4b': -15, '66d8': -14, '4e1e': -13,
          '29fe': -12, 'fe9b': -11, '1b0f': -10, '252e': -9, 'a8d2': -8, '7468': -7, '596a': -6, '47c1': -5, '0267': -4,
          'b314': -3, '5d7b': -2, '6bb6': -1, 'cfcd': 0, 'c4ca': 1, 'c81e': 2, 'eccb': 3, 'a87f': 4, 'e4da': 5,
          '1679': 6, '8f14': 7, 'c9f0': 8, '45c4': 9, 'd3d9': 10, '6512': 11, 'c20a': 12, 'c51c': 13, 'aab3': 14,
          '9bf3': 15, 'c74d': 16, '70ef': 17, '6f49': 18, '1f0e': 19, '98f1': 20}

zb2_post_map = {v: k for k, v in zb2map.items()}
print(zb2_post_map)
int_value = int(np.clip(np.round(30), -20, 20))
msg = zb2_post_map[int_value]
print(msg)
