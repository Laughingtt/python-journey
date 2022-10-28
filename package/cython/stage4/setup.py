# setup.py
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    name='hash_code',
    sources=['hash_code.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
)

setup(name="core_ml",
      version='1.5.1',
      description='corp',
      license='Apache',
      packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
      include_package_data=True,
      python_requires='>=3.6.0',
      install_requires=[
          'numpy'
      ],
      extras_require={},
      ext_modules=cythonize(extension),
      zip_safe=False)
