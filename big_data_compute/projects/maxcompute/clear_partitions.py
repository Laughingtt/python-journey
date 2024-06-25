from datetime import datetime, timedelta


def is_last_day_of_month(date_object):
    # 获取当月最后一天的日期
    last_day_of_month = (date_object.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

    # 判断日期是否为当月最后一天
    is_last_day = date_object.day == last_day_of_month.day

    return is_last_day


def del_table_partition(table_name, remain_day=30):
    ds_list = "select ds from ${table_name} order by ds"

    for ds_index, _ds in enumerate(ds_list):
        date_object = datetime.strptime(str(_ds), '%Y%m%d')

        # 当月最后一天保留
        if is_last_day_of_month(date_object):
            print("ds: {} 为当月最后一天，保留分区数据".format(_ds))
            continue

        # 近30天分区保留
        if (datetime.now() - date_object).days <= remain_day:
            continue

        # 列表最后分区跳过
        if ds_index >= len(ds_list) - 1:
            continue

        # 对比分区数据一致，删除此分区
        ds_add1 = ds_list[ds_index + 1]
        left_count = "select * from ${table_name} where ds=${ds} except select * from ${table_name} where ds=${ds_add1};"
        right_count = "select * from ${table_name} where ds=${ds_add1} except select * from ${table_name} where ds=${ds};"

        if left_count == 0 and right_count == 0:
            del_ = "alter table ${table_name} drop partition (ds=${_ds}) purge;"


def clear_tables(table, remain_day):
    tables = []
    for table in tables:
        del_table_partition(table, remain_day)
        print("table {} has clean partitions".format(table))


if __name__ == '__main__':
    clear_tables([], 30)
