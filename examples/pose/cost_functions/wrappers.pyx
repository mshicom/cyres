from cyres cimport CostFunction, LocalParameterization
cimport ceres
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref

np.import_array()

from eigency.core cimport *

cdef class SimilarityCost(CostFunction):
    def __cinit__(self):
        self._cost_function = _SimilarityCost.create()

cdef class CameraOdometerError(CostFunction):
    def __init__(self, np.ndarray x, np.ndarray y):
        x = np.asfortranarray(x)
        y = np.asfortranarray(y)
        self._cost_function = _CameraOdometerError.create(Map[Matrix4d](x),
                                                          Map[Matrix4d](y))
cdef extern from "Eigen/Geometry" namespace "Eigen":
    cppclass Quaterniond(PlainObjectBase):
        Quaterniond()
        Quaterniond(double *ptr)
        Quaterniond(Map[Matrix3d]& rmatrix)
        Quaterniond(Map[Vector4d]& rmatrix)
        Matrix3d toRotationMatrix()
        Vector4d coeffs()
        double* data"coeffs().data"()

cdef SO3FromQuat(const Quaterniond& q):
    return ndarray_copy( q.toRotationMatrix() )

cdef Quaterniond QuatFromSO3(np.ndarray so3):
    so3_f = np.asfortranarray(so3)
    return Quaterniond(Map[Matrix3d](so3_f))

def mockR(np.ndarray so3):
    return SO3FromQuat(QuatFromSO3(so3))

cdef Quaterniond QuatFromVec(np.ndarray q_coefs):
    cdef double[:] data = q_coefs.ravel()
    return Quaterniond(<double*>(&data[0]))


cdef object VecFromQuat(const Quaterniond& q):
    return ndarray(q.coeffs())

def QCoefFromSO3(np.ndarray so3):
    return VecFromQuat(QuatFromSO3(so3))

def SO3FromQCoef(np.ndarray q_coefs):
    return SO3FromQuat(QuatFromVec(q_coefs[:4]))



cdef extern from "cost_functions.h":
    cppclass _SimilarityCost "SimilarityCost"(ceres.CostFunction):
        @staticmethod
        ceres.CostFunction* create()

    cppclass _CameraOdometerError "CameraOdometerError"(ceres.CostFunction):
        @staticmethod
        ceres.CostFunction* create(Map[Matrix4d]& x,  Map[Matrix4d]& y)
