#! /usr/bin/env python
from distutils.core import *
from distutils      import sysconfig
import numpy


os.environ['CC'] = 'g++';

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# inplace extension module
_inplace = Extension("_PyiSDR",
                   ["PyiSDR.i","pywrapper.c"],
                   include_dirs = [numpy_include,"../inc","/home/bbelaouc/Wokspace/Needed/FLENS","/home/bbelaouc/anaconda2/lib/pkgconfig","/home/bbelaouc/anaconda2/lib/python2.7/config"],
                   library_dirs=['/home/bbelaouc/Wokspace/Needed/hdf5/hdf5/lib/', '/home/bbelaouc/Wokspace/Needed/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/',
                   '/home/bbelaouc/Wokspace/Needed/intel/mkl/lib/intel64_lin -L/usr/lib64/'],
                   libraries=['matio', 'hdf5','mkl_intel_ilp64','mkl_gnu_thread','mkl_core','pthread','m','dl','mkl_rt','iomp5','stdc++']
                   )

# NumyTypemapTests setup
setup(  name        = "PyiSDR",
        description = "PyiSDR",

        author      = "BB",
        version     = "1.0",
        ext_modules = [_inplace]
        )
