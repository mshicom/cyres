#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension
import os
import numpy
import cyres
import eigency

ext_modules = [
    Extension(
        "wrappers",
        sources = ["cost_functions/wrappers.pyx"],
        define_macros=[('EIGEN_DEFAULT_TO_ROW_MAJOR', '1')],
        language="c++",
        include_dirs=["/usr/include/eigen3", "./Sophus/", numpy.get_include()]
                        + eigency.get_includes(include_eigen=False),
        cython_include_dirs=[cyres.get_cython_include()],
        libraries = ["ceres", "glog","cholmod","lapack","gomp"],
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
