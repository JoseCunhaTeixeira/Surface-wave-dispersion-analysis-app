import os
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

fftw_include_dir = '/usr/local/include'
lib_dir = '/usr/local/lib'


if not os.path.exists("./lib/"):
   os.makedirs("./lib/")


extensions = [
    Extension(name="lib.dispersion", 
                sources=["src/dispersion.pyx"],
                language="c++",
                # include_dirs=[fftw_include_dir],
                # library_dirs=[lib_dir],
                libraries=["fftw3"],
                extra_compile_args=["-O3", "-fopenmp"],
                extra_link_args=["-O3", "-fopenmp"]),
]


setup(
    name="dispersion",
    ext_modules=cythonize(extensions),
)


subprocess.run(['mv', './src/dispersion.cpp', './lib/'])
