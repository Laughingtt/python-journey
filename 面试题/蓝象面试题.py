#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： tian
# datetime： 2023/2/7 6:41 PM 
# ide： PyCharm


"""
# 题目1：格式化文件大小
# 输入：文件大小（Bytes），int类型
# 输出：人性化显示文件大小。
# 例子：
#   56 -> 56B
#   56215 -> 54.9KB
#   26353560 -> 25.1MB
#   12312312312 -> 11.5GB
# 要求至少支持B/KB/MB/GB/TB/PB几个级别

"""


def test_01(bytes_size):
    tmp_size = bytes_size
    type_dict = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB", 5: "PB"}
    type_index = 0
    while tmp_size > 1024:
        tmp_size = tmp_size / 1024
        type_index += 1

    return str(tmp_size) + type_dict[type_index]


"""
# 题目2：有效的括号
# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
# 有效字符串需满足：
# - 左括号必须用相同类型的右括号闭合。
# - 左括号必须以正确的顺序闭合。
# 示例：
# "()" -> true
# "()[]{}" -> true
# "(]" -> false
# "[()]{}" -> true
"""


def compare_str(left, right):
    str_dict = {"(": ")", "[": "]", "{": "}"}
    if str_dict.get(left) == right:
        return True
    else:
        return False


def test02(string_s):
    list_s = [i for i in string_s]

    res = []
    while len(list_s) > 0:
        back_tmp = list_s.pop()
        if len(res) <= 0:
            res.append(back_tmp)
        else:
            if compare_str(back_tmp, res[-1]):
                res.pop()
            else:
                res.append(back_tmp)

    if len(res) > 0:
        return False
    else:
        return True


if __name__ == '__main__':
    print(test_01(12312312312))
    print(test02("[()]{}"))
    print(test02("[()]{}]"))
