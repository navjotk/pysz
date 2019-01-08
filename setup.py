from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([Extension("pysz", 
                                       sources=["pysz.pyx"],
                                       include_dirs=['sz-2.0.2.0/sz/include'])])
)
