import csv


def generate_sql():
    print("""
Create Table CMP_REPLACE_NAME(
      id int AUTO_INCREMENT COMMENT "原公司名称",
      input_cmp varchar(250) COMMENT "原公司名称",
      replace_cmp varchar(250) COMMENT "替换的公司名称",
      created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      primary key (id)
);
    """)
    print("INSERT INTO CMP_REPLACE_NAME(input_cmp, replace_cmp)")
    print("values")
    with open("cmp_list.csv", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        line = 0
        for row in csv_reader:
            line += 1
            if line == 1:
                continue
            print("('{}','{}'),".format(row[0].strip(), row[1].strip()))
            line += 1


if __name__ == '__main__':
    generate_sql()
