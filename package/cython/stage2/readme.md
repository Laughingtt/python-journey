### pxd 文件声明变量,cimport 进行导入


    重要的是要了解该cimport语句只能用于导入 C 数据类型、C 函数和变量以及扩展类型。它不能用于导入任何 Python 对象，并且（有一个例外）它在运行时并不意味着任何 Python 导入。如果要从已导入的模块中引用任何 Python 名称，则还必须为它包含一个常规的 import 语句。
    
    例外情况是，当您使用cimport导入扩展类型时，它的类型对象在运行时导入并通过导入时使用的名称提供。cimport下面更详细地介绍了如何使用来导入扩展类型。
    
    如果.pxd文件发生更改，cimport则可能需要重新编译来自该文件的任何模块。该Cython.Build.cythonize实用程序可以为您解决这个问题。

### 搜索路径

    当你cimport调用一个模块时modulename，Cython 编译器会搜索一个名为modulename.pxd. 它沿着包含文件的路径搜索此文件（由-I命令行选项或include_path 选项指定cythonize()），以及sys.path.
    
    使用package_data安装.pxd在你的文件setup.py脚本允许从模块依赖其他包cimport项目。
    
    此外，无论何时编译文件modulename.pyx，modulename.pxd都会首先沿着包含路径（而不是sys.path）搜索相应的定义文件，如果找到，则在处理.pyx文件之前对其进行处理。

python setup.py build_ext --inplace