import binascii

# 导入国密算法
from gmssl import sm2, sm4, sm3


def sm3_hash_(message: bytes):
    """
    国密sm3加密
    :param message: 消息值，bytes类型
    :return: 哈希值
    """

    hash_hex = sm3.sm3_hash(list(message))
    print(hash_hex)

    hash_bytes = bytes.fromhex(hash_hex)
    print(hash_bytes)

    bytes2hex(hash_bytes)

    # return bytes.hash
    # return hash


def bytes2hex(bytesData):
    hex = binascii.hexlify(bytesData)
    print(hex)
    print(hex.decode())
    return hex


def sm3_hash(message):
    return sm3.sm3_hash(list(bytes(message, encoding='utf-8')))


# main
if __name__ == '__main__':
    # print("main begin")
    # message = b"123456"  # bytes类型
    # sm3_hash_(message)
    print(sm3_hash("hello"))
