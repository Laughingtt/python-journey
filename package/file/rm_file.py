import os
import shutil

# 定义要删除的文件夹路径
folder_path = "/Users/tianjian/Projects/python-BasicUsage2/package/file/abc"

# 遍历文件夹中的所有文件并删除它们
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('删除 %s 失败，原因: %s' % (file_path, e))

# 创建新的空文件夹
# os.mkdir(folder_path)
