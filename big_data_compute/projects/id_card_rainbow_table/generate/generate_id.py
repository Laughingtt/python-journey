from data_driven_computing.id_card_rainbow_table.generate.conf.constant_area_2022 import AREA_INFO
from data_driven_computing.id_card_rainbow_table.generate.conf.constant import START_DATE, END_DATE, SEQUENCE_CODE
import datetime
import numpy as np


class GenerateFlow:
    def __init__(self):
        pass

    def run(self):
        generator_area_code = self.get_area_id()
        sample_are_code = next(generator_area_code)
        generator_birth_days = self.get_birth_date()

        _count = 0
        for birth_day in generator_birth_days:
            area_birth_id = str(sample_are_code[0]) + str(birth_day)
            # print("id code :{}-{}".format(area_birth_id, sample_are_code[1]))

            for seq in SEQUENCE_CODE[:10]:
                id_code = area_birth_id + seq
                id_code += self.get_check_digit(id_code)
                print("id code :{}-{}".format(id_code, sample_are_code[1]))
                _count += 1
        print("total id code is {}".format(_count))
        # self.get_sequence_code()

    def get_area_id(self):
        """
        省份，城市，区县码
            1-6 位
        """
        for area_code, area_name in AREA_INFO.items():
            yield area_code, area_name

    def get_birth_date(self):
        """
        出生日期
            7-14位
        """
        days = (datetime.datetime.strptime(END_DATE, "%Y-%m-%d") -
                datetime.datetime.strptime(START_DATE, "%Y-%m-%d")).days + 1

        start_strptime = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
        for day_num in range(days):
            birth_days = datetime.datetime.strftime(
                start_strptime + datetime.timedelta(day_num), "%Y%m%d")
            yield birth_days

    def get_birth_year(self):
        """
        出生年
            7-10
        """
        pass

    def get_birth_month(self):
        """
        出生月
            11-12
        """
        pass

    def get_birth_day(self):
        """
        出生日
            13-14
        """
        pass

    def get_sequence_code(self):
        """
        顺序码
            第15-17位都是同一地址辖区内的，以及同年同月同日出生人的顺序码，
            同时第17位兼具性别标识功能，男单女双

        """
        sequence = []
        for seq in range(1, 1000):
            if seq < 10:
                sequence.append("00" + str(seq))
            elif seq < 100:
                sequence.append("0" + str(seq))
            else:
                sequence.append(str(seq))
        print(sequence)

    def get_sex(self):
        """
        性别
            第17位
            男 是奇数，女是偶数
        """
        pass

    def get_check_digit(self, Id):
        """
        校验码
            第18位数字是校检码：可以是0-9的数字，有时也用X表示10
        校验码计算公式
            1. 将身份证号码从右至左标记为，a1,a2,a3,......a18, a1即为校验码
            2. 计算权重系数 W_i = 2**(i-1) % 11
                i	18	17	16	15	14	13	12	11	10	9	8	7	6	5	4	3	2	1
                Wi	7	9	10	5	8	4	2	1	6	3	7	9	10	5	8	4	2	1
            3. S = np.sum(a_list[1:] * W_list[1:])
            4. a1 = (12-(S % 11)) % 11

        示例:
            输入:
                "41032619950710319"
            输出:
                7

        """
        weight = np.array([1, 2, 4, 8, 5, 10, 9, 7, 3, 6, 1, 2, 4, 8, 5, 10, 9, 7], dtype=np.int8)[1:]
        id_code = np.flipud(np.array(list(Id), dtype=np.int8))
        s_const = np.sum(weight * id_code)
        check_digit = (12 - (s_const % 11)) % 11
        return str(check_digit) if check_digit < 10 else 'X'


if __name__ == '__main__':
    g = GenerateFlow()
    g.run()

