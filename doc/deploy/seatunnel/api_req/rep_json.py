import os
import json

# 读取原始 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 环境变量
os.environ['username'] = 'myuser'
os.environ['password'] = 'mypassword'

# 替换 JSON 字符串中的变量
def replace_vars_in_json(template: str):
    replaced_template = template
    for key, value in os.environ.items():
        replaced_template = replaced_template.replace(f'${{{key}}}', value)
    return replaced_template

# 示例：原始 JSON 文件路径
input_file = 'template.json'  # 你的原始 JSON 文件路径

# 读取原始 JSON 文件
json_content = read_json_file(input_file)

# 替换模板中的变量
replaced_json = replace_vars_in_json(json_content)

# 将替换后的 JSON 字符串转换为字典
replaced_dict = json.loads(replaced_json)


print(f"替换后的 JSON  {replaced_dict}")
