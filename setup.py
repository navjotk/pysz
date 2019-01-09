from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("pysz", 
                                       sources=["pysz.pyx"],
                                       include_dirs=['sz-2.0.2.0/sz/include'],
                            libraries=["SZ", "zlib", "zstd"],  # Unix-like specific,
              library_dirs=["sz-2.0.2.0/lib"],
              extra_compile_args=['-fopenmp', '-g'],
             extra_link_args=['-fopenmp', '-g', '-Wl,-rpath,/usr/local/lib'])]
setup(name="pysz",
    ext_modules = cythonize(ext_modules, gdb_debug=True))
