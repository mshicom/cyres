from cyres cimport CostFunction, LocalParameterization
cimport ceres
cimport numpy
import numpy as np
from cython.operator cimport dereference as deref

numpy.import_array()
from eigency.core cimport *


cdef double* getDoublePtr(object param):
    cdef double[::1] data = (<np.ndarray?>param).reshape(-1)
    return <double*>(&data[0])

cdef object Mat1DFromPtr(double* ptr, int size):
    return np.asarray(<double[:size]> ptr)

cdef class LocalParameterizationSE3(LocalParameterization):
    def __cinit__(self):
        self._local_parameterization = new _LocalParameterizationSE3()

cdef class TestCostFunctor(CostFunction):
    def __cinit__(self, SE3 T_aw):
        self._cost_function = _TestCostFunctor.create(T_aw.ptr_[0])

cdef extern from "sophus_cost_functions.h":
    cppclass _TestCostFunctor "TestCostFunctor"(ceres.CostFunction):
        @staticmethod
        ceres.CostFunction* create(SE3d T_aw)

    cppclass _LocalParameterizationSE3 "LocalParameterizationSE3"(ceres.LocalParameterization):
        pass


cdef extern from "sophus/se3.hpp" namespace "Sophus":
    cppclass SE3d:
        SE3d()
        SE3d(const SE3d& other)
        SE3d(const Matrix4d& T)
        void normalize()
        double* data()
        SE3d inverse()
        SE3d operator*(const SE3d& other) const
        Vector3d operator*(const Vector3d& p) const
        SE3d& operator=(const SE3d& other)

        PlainObjectBase Adj() const
        PlainObjectBase log() const
        PlainObjectBase internalMultiplyByGenerator(int i) const
        PlainObjectBase internalJacobian() const
        PlainObjectBase matrix() const
        PlainObjectBase matrix3x4() const
        PlainObjectBase rotationMatrix() const
        PlainObjectBase translation()
        @staticmethod
        SE3d exp(const Vector6d& a)

    cppclass Vector6d(PlainObjectBase):
        pass
    cppclass Matrix6d(PlainObjectBase):
        pass

cdef extern from "sophus/se3.hpp" namespace "Eigen":
    cppclass SE3Map "Map<Sophus::SE3d>"(SE3d):
        SE3Map(const double* coeffs)
        SE3Map(const double* trans_coeffs, const double* rot_coeffs)

cdef class SE3:
    cdef:
        SE3d* ptr_
        bint isSelfOwned
        np.ndarray data_array
    def __init__(self, isSelfOwned = False):
        self.isSelfOwned = isSelfOwned
        self.data_array = None

    def __dealloc__(self):
        if self.isSelfOwned:
            del self.ptr_

    cpdef inverse(self):
        return warpSE3dResult(self.ptr_.inverse())

    cpdef Adj(self):
        return ndarray_copy(self.ptr_.Adj())

    cpdef log(self):
        return ndarray_copy(self.ptr_.log())

    cpdef internalMultiplyByGenerator(self, int i):
        return ndarray_copy(self.ptr_.internalMultiplyByGenerator(i))

    cpdef internalJacobian(self):
        return ndarray_copy(self.ptr_.internalJacobian())

    cpdef matrix(self):
        return ndarray_copy(self.ptr_.matrix())

    cpdef matrix3x4(self):
        return ndarray_copy(self.ptr_.matrix3x4())

    cpdef rotationMatrix(self):
        return ndarray_copy(self.ptr_.rotationMatrix())

    cpdef normalize(self):
        self.ptr_.normalize()

    cpdef assign(self, SE3 other):
        self.ptr_[0] = other.ptr_[0]

    cpdef copy(self):
        return warpSE3dResult(self.ptr_[0])

    def __mul__(SE3 a, b):
        if isinstance(b, SE3):
            return warpSE3dResult(a.ptr_[0] * (<SE3>b).ptr_[0])
        else:
            return ndarray_copy(a.ptr_[0] * <Vector3d>Map[Vector3d](b))

    def __repr__(self):
        return str(self.matrix())

    property data:
        def __get__(self):
            if self.data_array is None:
                self.data_array = Mat1DFromPtr(self.ptr_.data(), 7)
            return self.data_array

    property translation:
        def __get__(self):                      return self.data[4:]
        def __set__(self, double[::1] value):   self.data[4:] = value[:3]

    @classmethod
    def create(cls, np.ndarray T=np.eye(4)):
        ret = SE3(isSelfOwned=True)
        T =  np.asfortranarray(T)
        ret.ptr_ = new SE3d(<Matrix4d>Map[Matrix4d](T))
        return ret

    @staticmethod
    cdef warpPtr(SE3d *ptr):
        if ptr!= NULL:
            T_ = SE3(isSelfOwned=False)
            T_.ptr_ = ptr
            return T_
        else:
            return None

    @staticmethod
    def exp(np.ndarray vec6):
        return warpSE3dResult(SE3d.exp(<Vector6d>Map[Vector6d](vec6)))


cdef warpSE3dResult(const SE3d &obj):
    ret = SE3(isSelfOwned=True)
    ret.ptr_ = new SE3d(obj)
    return ret

