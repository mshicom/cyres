#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension
import os
import numpy
import cyres
import eigency

include_dirs = ["/usr/include/eigen3", numpy.get_include()]+ eigency.get_includes(include_eigen=False)
libraries = ["ceres", "glog","cholmod","lapack","gomp"]
ext_modules = [
    Extension(
        "wrappers",
        sources = ["cost_functions/wrappers.pyx"],
        define_macros=[('SOPHUS_CERES_FOUND', '1')],
        language="c++",
        include_dirs= include_dirs,
        cython_include_dirs=[cyres.get_cython_include()],
        libraries = libraries,
        extra_compile_args = ["-std=c++11"]
    ),

    Extension(
        "sophus",
        define_macros=[('SOPHUS_CERES_FOUND', '1')],
        sources = ["cost_functions/sophus.pyx"],
        language="c++",
        include_dirs= include_dirs,
        cython_include_dirs=[cyres.get_cython_include()],
        libraries = libraries,
        extra_compile_args = ["-std=c++11"]
    )
]

setup(
  name = 'cost_functions',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cost_functions',
  ext_modules = ext_modules,
)
