from cyres cimport CostFunction, LocalParameterization
cimport ceres
cimport numpy
import numpy as np
from cython.operator cimport dereference as deref

numpy.import_array()


cdef class SimilarityCost(CostFunction):
    def __cinit__(self):
        self._cost_function = _SimilarityCost.create()

cdef extern from "cost_functions.h":
    cppclass _SimilarityCost "SimilarityCost"(ceres.CostFunction):
        @staticmethod
        ceres.CostFunction* create()

