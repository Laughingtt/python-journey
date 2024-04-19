from datetime import datetime, timedelta


def is_last_day_of_month(date_list):
    result = []
    for date_str in date_list:
        # 将日期字符串转换为datetime对象
        date_object = datetime.strptime(str(date_str), '%Y%m%d')

        # 获取当月最后一天的日期
        last_day_of_month = (date_object.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

        # 判断日期是否为当月最后一天
        is_last_day = date_object.day == last_day_of_month.day
        result.append(is_last_day)

    return result


date_list = [20201113, 20211130, 20211129, 20231230,20231231]
result = is_last_day_of_month(date_list)
print(result)
