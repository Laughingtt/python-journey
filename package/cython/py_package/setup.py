from setuptools import setup, find_packages

setup(
    name='printtest',
    version='1.0',
    description='hello world',
    license='Apache',
    author="tian",
    author_email="tianjian361@163.com",
    include_package_data=True,
    # 自动包含包内所有受版本控制(cvs/svn/git)的数据文件

    packages=find_packages(include=["pro", "pro.*"]),
    # 需要处理的包目录（包含__init__.py的文件夹）和setup.py同一目录
    # 下搜索各个含有 init.py的包,也可以指定find_packages(),代表打包所有文件

    package_data={'': ['*.json', '*.csv']},
    # 也可以用做打包非py文件，可以使用正则匹配的方式,但文件目录必须包含__init__.py

    data_files=[('pro', ['pro/data/t.json', 'pro/data/t.csv'])],
    # 打包时非py文件存在时,必须得具体指定某个文件的相对路径


    python_requires='>=3.6.0',
    install_requires=['decorator==4.3.0'],
    # 定义依赖哪些模块 如果不存在自动下载，存在则跳过

    zip_safe=False,
    extras_require={}
)
