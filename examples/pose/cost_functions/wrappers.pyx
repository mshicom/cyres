from cyres cimport CostFunction, LocalParameterization
cimport ceres
cimport numpy as np
import numpy as np

np.import_array()

cdef class LocalParameterizationSE3(LocalParameterization):
    def __cinit__(self):
        self._local_parameterization = new _LocalParameterizationSE3()

cdef class SimilarityCost(CostFunction):

    def __cinit__(self):
        self._cost_function = create()

cdef extern from "cost_functions.h":
    ceres.CostFunction* create "SimilarityCost::create"()
    cppclass _LocalParameterizationSE3 "LocalParameterizationSE3"(ceres.LocalParameterization):
        pass


from eigency.core cimport *
cdef extern from "sophus/se3.hpp" namespace "Sophus":
    cppclass SE3d:
        SE3d()
        SE3d(const SE3d& other)
        SE3d(const Matrix4d& T)
        Matrix4d matrix()
        double* data()
#        const double* data()

cdef extern from "sophus/se3.hpp" namespace "Eigen":
    cppclass SE3Map "Map<Sophus::SE3d>"(SE3d):
        SE3Map(const double* coeffs)
        SE3Map(const double* trans_coeffs, const double* rot_coeffs)

cdef class SE3:
    cdef:
        SE3d* ptr_
        bint isSelfOwned
    def __init__(self, isSelfOwned = False):
        self.isSelfOwned = isSelfOwned
    def __dealloc__(self):
        if self.isSelfOwned:
            del self.ptr_
    def matrix(self):
        return ndarray_copy(self.ptr_.matrix())

    def data(self):
        cdef np.npy_intp dims[1]
        dims[0] = 7
        return np.PyArray_SimpleNewFromData(1, <np.npy_intp*>dims,
                                            np.NPY_FLOAT64,
                                            <void*>(self.ptr_.data()))

    @classmethod
    def create(cls, np.ndarray T):
        T_ = SE3(isSelfOwned=True)
        T_.ptr_ = new SE3d(<Matrix4d>Map[Matrix4d](T))
        return T_

    @staticmethod
    cdef warpPtr(SE3d *ptr):
        if ptr!= NULL:
            T_ = SE3()
            T_.ptr_ = ptr
            return T_
        else:
            return None




#cdef extern from "Eigen/Eigen" namespace "Eigen":
#    cppclass Vector6d(PlainObjectBase):
#        pass
#    cppclass Matrix6d(PlainObjectBase):
#        pass
#
#
#cdef extern from "sophus/so3.hpp" namespace "Sophus":
#    cppclass SO3d:
#        pass
#
#cdef extern from "sophus/se3.hpp" namespace "Sophus":
#    int DoF = 6            # degree of freedom
#    int num_parameters = 7 # number of internal parameters
#    int N = 4              # NxN matrices
#    ctypedef Matrix4d Transformation
#    ctypedef Vector3d Point
#    ctypedef Vector6d Tangent
#    ctypedef Matrix6d Adjoint
#
#    cppclass SE3GroupBase[T]:
#        Adjoint Adj() const
##        Eigen::Transform<Scalar, 3, Eigen::Affine> affine3() const
#        U cast[U]() const
##        Eigen::Matrix<Scalar, num_parameters, 1>  internalMultiplyByGenerator(int i) const
##        Eigen::Matrix<Scalar, num_parameters, DoF>  internalJacobian() const
#        SE3Group inverse() const
#        Tangent log() const
#        void normalize()
#        Transformation matrix() const
##        Eigen::Matrix<Scalar, 3, 4> matrix3x4()  const
#        U& operator=[U](const U& other)
##        SE3GroupBase operator*(const SE3d& other)
##        Point operator*(const Point& p) const
##        T& operator*=[T](const T& other)
##        Matrix3d rotationMatrix() const
##        SO3d& so3()
##        PlainObjectBase& translation()
##        SE3GroupBase exp(const Tangent& a)
##
##        Transformation hat(const Tangent& v)
#
#    cppclass SE3Group(SE3GroupBase):
#        SE3Group()
##        SE3Group[U](const SE3GroupBase<OtherDerived>& other)

