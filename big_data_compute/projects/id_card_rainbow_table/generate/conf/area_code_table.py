def read_area_code(dir):
    dic = {}
    with open(dir, "r", encoding='utf-8') as f:
        data = f.readlines()
        for i in data:
            sub_d = i.split("\t")
            dic[sub_d[0].strip()] = sub_d[1].strip()

    print(len(dic))
    print(dic)


if __name__ == '__main__':
    read_area_code(
        r'C:\Users\TianJian\Desktop\Projects\python-journey\data_driven_computing\id_card_rainbow_table\generate\2022区域代码表.md')
