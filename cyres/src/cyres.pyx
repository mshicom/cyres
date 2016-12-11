#cython: boundscheck=False, wraparound=False

import cython
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as drf
cimport cpython.ref as cpy_ref


cimport numpy as np

import numpy as np
np.import_array()

cimport ceres
from cyres cimport *

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

Ownership = enum("DO_NOT_TAKE_OWNERSHIP", "TAKE_OWNERSHIP")
MinimizerType = enum("LINE_SEARCH", "TRUST_REGION")
LinearSolverType = enum("DENSE_NORMAL_CHOLESKY", "DENSE_QR",
                        "SPARSE_NORMAL_CHOLESKY", "DENSE_SCHUR", "SPARSE_SCHUR",
                        "ITERATIVE_SCHUR", "CGNR")
PreconditionerType = enum("IDENTITY", "JACOBI", "SCHUR_JACOBI",
                          "CLUSTER_JACOBI", "CLUSTER_TRIDIAGONAL")
SparseLinearAlgebraLibraryType = enum("SUITE_SPARSE", "CX_SPARSE")
LinearSolverTerminationType = enum("TOLERANCE", "MAX_ITERATIONS", "STAGNATION",
                                   "FAILURE")
LoggingType = enum("SILENT", "PER_MINIMIZER_ITERATION")
LineSearchDirectionType = enum("STEEPEST_DESCENT",
                               "NONLINEAR_CONJUGATE_GRADIENT",
                               "LBFGS")
NonlinearConjugateGradientType = enum("FLETCHER_REEVES", "POLAK_RIBIRERE",
                                      "HESTENES_STIEFEL")
LineSearchType = enum("ARMIJO")
TrustRegionStrategyType = enum("LEVENBERG_MARQUARDT", "DOGLEG")
DoglegType = enum("TRADITIONAL_DOGLEG", "SUBSPACE_DOGLEG")
SolverTerminationType = enum("DID_NOT_RUN", "NO_CONVERGENCE", "FUNCTION_TOLERANCE", "GRADIENT_TOLERANCE", "PARAMETER_TOLERANCE", "NUMERICAL_FAILURE", "USER_ABORT", "USER_SUCCESS")
CallbackReturnType = enum("SOLVER_CONTINUE", "SOLVER_ABORT", "SOLVER_TERMINATE_SUCCESSFULLY")
DumpFormatType = enum("CONSOLE", "PROTOBUF", "TEXTFILE")
DimensionType = enum(DYNAMIC=-1)
NumericDiffMethod = enum("CENTRAL", "FORWARD")

cdef class CostFunction:

    cpdef parameter_block_sizes(self):
        block_sizes = []
        cdef vector[ceres.int32] _parameter_block_sizes = self._cost_function.parameter_block_sizes()
        for i in range(_parameter_block_sizes.size()):
            block_sizes.append(_parameter_block_sizes[i])
        return block_sizes

    cpdef num_residuals(self):
        return self._cost_function.num_residuals()

    def evaluate(self, *param_blocks, **kwargs):

        include_jacobians = kwargs.get("include_jacobians", False)

        cdef double** _params_ptr = NULL
        cdef double* _residuals_ptr = NULL
        cdef double** _jacobians_ptr = NULL

        block_sizes = self.parameter_block_sizes()

        _params_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))

        cdef np.ndarray[np.double_t, ndim=1] _param_block

        for i, param_block in enumerate(param_blocks):
            if block_sizes[i] != len(param_block):
                raise Exception("Expected param block of size %d, got %d" % (block_sizes[i], len(param_block)))
            _param_block = param_block
            _params_ptr[i] = <double*> _param_block.data

        cdef np.ndarray[np.double_t, ndim=1] residuals

        residuals = np.empty((self.num_residuals()), dtype=np.double)
        _residuals_ptr = <double*> residuals.data

        cdef np.ndarray[np.double_t, ndim=2] _jacobian
        if include_jacobians:
            # jacobians is an array of size CostFunction::parameter_block_sizes_
            # containing pointers to storage for Jacobian matrices corresponding
            # to each parameter block. The Jacobian matrices are in the same
            # order as CostFunction::parameter_block_sizes_. jacobians[i] is an
            # array that contains CostFunction::num_residuals_ x
            # CostFunction::parameter_block_sizes_[i] elements. Each Jacobian
            # matrix is stored in row-major order, i.e., jacobians[i][r *
            # parameter_block_size_[i] + c]
            jacobians = []
            _jacobians_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))
            for i, block_size in enumerate(block_sizes):
                jacobian = np.empty((self.num_residuals(), block_size), dtype=np.double)
                jacobians.append(jacobian)
                _jacobian = jacobian
                _jacobians_ptr[i] = <double*> _jacobian.data

        self._cost_function.Evaluate(_params_ptr, _residuals_ptr, _jacobians_ptr)

        free(_params_ptr)

        if include_jacobians:
            free(_jacobians_ptr)
            return residuals, jacobians
        else:
            return residuals

cdef class SquaredLoss(LossFunction):
    def __cinit__(self):
        _loss_function = NULL

cdef class HuberLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.HuberLoss(_a)

cdef class SoftLOneLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.SoftLOneLoss(_a)

cdef class CauchyLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.CauchyLoss(_a)

cdef class ArctanLoss(LossFunction):
    def __init__(self, double _a):
        """
        Loss that is capped beyond a certain level using the arc-tangent
        function. The scaling parameter 'a' determines the level where falloff
        occurs. For costs much smaller than 'a', the loss function is linear
        and behaves like TrivialLoss, and for values much larger than 'a' the
        value asymptotically approaches the constant value of a * PI / 2.

          rho(s) = a atan(s / a).

        At s = 0: rho = [0, 1, 0].
        """
        self._loss_function = new ceres.ArctanLoss(_a)

cdef class TolerantLoss(LossFunction):
    """
    Loss function that maps to approximately zero cost in a range around the
    origin, and reverts to linear in error (quadratic in cost) beyond this
    range. The tolerance parameter 'a' sets the nominal point at which the
    transition occurs, and the transition size parameter 'b' sets the nominal
    distance over which most of the transition occurs. Both a and b must be
    greater than zero, and typically b will be set to a fraction of a. The
    slope rho'[s] varies smoothly from about 0 at s <= a - b to about 1 at s >=
    a + b.

    The term is computed as:

      rho(s) = b log(1 + exp((s - a) / b)) - c0.

    where c0 is chosen so that rho(0) == 0

      c0 = b log(1 + exp(-a / b)

    This has the following useful properties:

      rho(s) == 0               for s = 0
      rho'(s) ~= 0              for s << a - b
      rho'(s) ~= 1              for s >> a + b
      rho''(s) > 0              for all s

    In addition, all derivatives are continuous, and the curvature is
    concentrated in the range a - b to a + b.

    At s = 0: rho = [0, ~0, ~0].
    """
    def __init__(self, double _a, double _b):
        self._loss_function = new ceres.TolerantLoss(_a, _b)

cdef class ComposedLoss(LossFunction):
    def __init__(self, LossFunction f, LossFunction g):
        self._loss_function = new ceres.ComposedLoss(f._loss_function,
                                                     ceres.DO_NOT_TAKE_OWNERSHIP,
                                                     g._loss_function,
                                                     ceres.DO_NOT_TAKE_OWNERSHIP)

cdef class ScaledLoss(LossFunction):
    def __init__(self, LossFunction loss_function, double _a):
        self._loss_function = new ceres.ScaledLoss(loss_function._loss_function,
                                                   _a,
                                                   ceres.DO_NOT_TAKE_OWNERSHIP)

cdef class Summary:
    cdef ceres.Summary _summary

    def briefReport(self):
        return self._summary.BriefReport()

    def fullReport(self):
        return self._summary.FullReport()

cdef class EvaluateOptions:
    cdef ceres.EvaluateOptions _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = ceres.EvaluateOptions()

    property residual_blocks:
        def __get__(self):
            return [warpResidualBlockId(block) for block in self._options.residual_blocks]
        def __set__(self, blocks):
            self._options.residual_blocks.clear()
            for block in blocks:
                self._options.residual_blocks.push_back((<ResidualBlockId?>block)._block_id)

    property apply_loss_function:
        def __get__(self):        return self._options.apply_loss_function
        def __set__(self, value): self._options.apply_loss_function = value

cdef class SolverOptions:
    cdef ceres.SolverOptions* _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = new ceres.SolverOptions()

    property max_num_iterations:
        def __get__(self):        return self._options.max_num_iterations
        def __set__(self, value): self._options.max_num_iterations = value

    property minimizer_progress_to_stdout:
        def __get__(self):        return self._options.minimizer_progress_to_stdout
        def __set__(self, value): self._options.minimizer_progress_to_stdout = value

    property linear_solver_type:
        def __get__(self):        return self._options.linear_solver_type
        def __set__(self, value): self._options.linear_solver_type = value

    property trust_region_strategy_type:
        def __get__(self):        return self._options.trust_region_strategy_type
        def __set__(self, value): self._options.trust_region_strategy_type = value

    property dogleg_type:
        def __get__(self):        return self._options.dogleg_type
        def __set__(self, value): self._options.dogleg_type = value

    property preconditioner_type:
        def __get__(self):        return self._options.preconditioner_type
        def __set__(self, value): self._options.preconditioner_type = value

    property num_threads:
        def __get__(self):        return self._options.num_threads
        def __set__(self, value): self._options.num_threads = value

    property num_linear_solver_threads:
        def __get__(self):        return self._options.num_linear_solver_threads
        def __set__(self, value): self._options.num_linear_solver_threads = value

    property use_nonmonotonic_steps:
        def __get__(self):        return self._options.use_nonmonotonic_steps
        def __set__(self, value): self._options.use_nonmonotonic_steps = value

    property gradient_tolerance:
        def __get__(self):        return self._options.gradient_tolerance
        def __set__(self, value): self._options.gradient_tolerance = value

    property function_tolerance:
        def __get__(self):        return self._options.function_tolerance
        def __set__(self, value): self._options.function_tolerance = value

    property parameter_tolerance:
        def __get__(self):        return self._options.parameter_tolerance
        def __set__(self, value): self._options.parameter_tolerance = value

    def add_callback(self, callback):
        self._options.callbacks.push_back((<IterationCallback?>callback)._ptr)


cdef extern from "callback.h":
    cppclass PythonCallback(ceres.IterationCallback):
        PythonCallback(cpy_ref.PyObject* obj)
        ceres.CallbackReturnType operator()(const ceres.IterationSummary& summary)

cdef public api ceres.CallbackReturnType cy_callback(object obj, object summary):
    ''' this will be used in cpp file'''
    if not hasattr(obj, "__call__"):
        raise RuntimeError("no call method defined")
    ret = obj.__call__(summary)
    return <ceres.CallbackReturnType>ret

from collections import namedtuple
IterationSummary = namedtuple('IterationSummary',
                              ['iteration',
                               'step_is_valid',
                               'step_is_nonmonotonic',
                               'step_is_successful',
                               'cost',
                               'cost_change',
                               'gradient_max_norm',
                               'step_norm',
                               'relative_decrease',
                               'trust_region_radius',
                               'eta',
                               'step_size',
                               'line_search_function_evaluations',
                               'linear_solver_iterations',
                               'iteration_time_in_seconds',
                               'step_solver_time_in_seconds',
                               'cumulative_time_in_seconds'])

cdef public api object cy_warpSummary(const ceres.IterationSummary& summary):
    return IterationSummary(summary.iteration,
                           summary.step_is_valid,
                           summary.step_is_nonmonotonic,
                           summary.step_is_successful,
                           summary.cost,
                           summary.cost_change,
                           summary.gradient_max_norm,
                           summary.step_norm,
                           summary.relative_decrease,
                           summary.trust_region_radius,
                           summary.eta,
                           summary.step_size,
                           summary.line_search_function_evaluations,
                           summary.linear_solver_iterations,
                           summary.iteration_time_in_seconds,
                           summary.step_solver_time_in_seconds,
                           summary.cumulative_time_in_seconds)

cdef class IterationCallback:
    cdef ceres.IterationCallback* _ptr
    def __cinit__(self, *args, **kw):
        self._ptr = new PythonCallback(<cpy_ref.PyObject*>self)
    def __dealloc__(self):
        if self._ptr!=NULL:
            del self._ptr
    def __call__(self, summary):
#        print summary
        return CallbackReturnType.SOLVER_CONTINUE

class SimpleCallback(IterationCallback):
    def __init__(self, callable_object):
        super().__init__(self)
        if not callable(callable_object):
            raise RuntimeError("not callable")
        self.func =  callable_object
    def __call__(self, summary):
        self.func()
        return CallbackReturnType.SOLVER_CONTINUE

""" This function will get the raw pointer of a numpy array that:
        1. can be of any shape,
        2. but with the type of double and
        3. being c-continous (after flattened if more than 1D)
    If requirements are not fullfilled then an Exception will be raised.
    It uses cython memoryview to do the checking.
"""
cdef double* getDoublePtr(object param):
    cdef double[::1] data = (<np.ndarray?>param).reshape(-1)
    return <double*>(&data[0])

cdef class Problem:
    cdef ceres.Problem _problem

    def __cinit__(self):
        pass

    # loss_function=NULL yields squared loss
    def add_residual_block(self,
                           CostFunction cost_function,
                           LossFunction loss_function,
                           *parameter_blocks):
        cdef vector[double*] _parameter_blocks
        cdef ceres.ResidualBlockId _block_id

        for parameter_block in parameter_blocks:
            _parameter_blocks.push_back(getDoublePtr(parameter_block))

        _block_id = self._problem.AddResidualBlock(cost_function._cost_function,
                                                   loss_function._loss_function,
                                                   _parameter_blocks)
        return warpResidualBlockId(_block_id)

    def evaluate(self, residual_blocks, apply_loss_function=True):
        cdef double cost

        options = EvaluateOptions()
        options.apply_loss_function = apply_loss_function
        options.residual_blocks = residual_blocks

        self._problem.Evaluate(options._options, &cost, NULL, NULL, NULL)
        return cost

    def set_parameter_block_constant(self, block):
        self._problem.SetParameterBlockConstant(getDoublePtr(block))

    def set_parameter_block_variable(self, block):
        self._problem.SetParameterBlockVariable(getDoublePtr(block))

    def add_parameter_block(self, block, int size, LocalParameterization lp=None):
        if lp is None:
            self._problem.AddParameterBlock(getDoublePtr(block), size)
        else:
            self._problem.AddParameterBlock(getDoublePtr(block), size,
                                            lp._local_parameterization)

    def set_parameterization(self, block, LocalParameterization lp):
        self._problem.SetParameterization(getDoublePtr(block),
                                          lp._local_parameterization)

    def set_parameter_lower_bound(self, block, int index, double lower_bound):
        self._problem.SetParameterLowerBound(getDoublePtr(block), index, lower_bound)

    def set_parameter_upper_bound(self, block, int index, double upper_bound):
        self._problem.SetParameterUpperBound(getDoublePtr(block), index, upper_bound)


cdef class ResidualBlockId:
    cdef ceres.ResidualBlockId _block_id

cdef object warpResidualBlockId(ceres.ResidualBlockId id_):
        RBid = ResidualBlockId()
        RBid._block_id = id_
        return RBid

def solve(SolverOptions options, Problem problem, Summary summary):
    ceres.Solve(drf(options._options), &problem._problem, &summary._summary)

cdef class IdentityParameterization(LocalParameterization):
    """ Identity Parameterization: Plus(x, delta) = x + delta"""
    def __init__(self, int size):
        self._local_parameterization = new ceres.IdentityParameterization(size)

cdef class SubsetParameterization(LocalParameterization):
    """ Hold a subset of the parameters inside a parameter block constant."""
    def __init__(self, int size, list constant_parameters):

        cdef vector[int] constant_parameters_
        for index in constant_parameters:
            constant_parameters_.push_back(<int?>(index))
        self._local_parameterization =                                  \
            new ceres.SubsetParameterization(size, constant_parameters_)

cdef class QuaternionParameterization(LocalParameterization):
    """
    Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
    with * being the quaternion multiplication operator. Here we assume
    that the first element of the quaternion vector is the real (cos
    theta) part.
    """
    def __init__(self):
        self._local_parameterization = new ceres.QuaternionParameterization()

cdef class EigenQuaternionParameterization(LocalParameterization):
    """
    Implements the quaternion local parameterization for Eigen's representation
    of the quaternion. Eigen uses a different internal memory layout for the
    elements of the quaternion than what is commonly used. Specifically, Eigen
    stores the elements in memory as [x, y, z, w] where the real part is last
    whereas it is typically stored first. Note, when creating an Eigen quaternion
    through the constructor the elements are accepted in w, x, y, z order. Since
    Ceres operates on parameter blocks which are raw double pointers this
    difference is important and requires a different parameterization.

    Plus(x, delta) = [sin(|delta|) delta / |delta|, cos(|delta|)] * x
    with * being the quaternion multiplication operator.
    """
    def __init__(self):
        self._local_parameterization = new ceres.EigenQuaternionParameterization()

cdef class HomogeneousVectorParameterization(LocalParameterization):
    """
    This provides a parameterization for homogeneous vectors which are commonly
    used in Structure for Motion problems.  One example where they are used is
    in representing points whose triangulation is ill-conditioned. Here
    it is advantageous to use an over-parameterization since homogeneous vectors
    can represent points at infinity.

    The plus operator is defined as
    Plus(x, delta) =
       [sin(0.5 * |delta|) * delta / |delta|, cos(0.5 * |delta|)] * x
    with * defined as an operator which applies the update orthogonal to x to
    remain on the sphere. We assume that the last element of x is the scalar
    component. The size of the homogeneous vector is required to be greater than
    1.
    """
    def __init__(self, int size):
        self._local_parameterization = new ceres.HomogeneousVectorParameterization(size)

cdef class ProductParameterization(LocalParameterization):
    """
    Construct a local parameterization by taking the Cartesian product
    of a number of other local parameterizations. This is useful, when
    a parameter block is the cartesian product of two or more
    manifolds. For example the parameters of a camera consist of a
    rotation and a translation, i.e., SO(3) x R^3.

    Currently this class supports taking the cartesian product of up to
    four local parameterizations.

    Example usage:

    ProductParameterization product_param(new QuaterionionParameterization(),
                                          new IdentityParameterization(3));

    is the local parameterization for a rigid transformation, where the
    rotation is represented using a quaternion.
    """
    def __init__(self, *args):
        # TODO Sanity Check
        if len(args)==2:
            self._local_parameterization = new ceres.ProductParameterization(
                (<LocalParameterization?>args[0])._local_parameterization,
                (<LocalParameterization?>args[1])._local_parameterization)

        elif len(args)==3:
            self._local_parameterization = new ceres.ProductParameterization(
                (<LocalParameterization?>args[0])._local_parameterization,
                (<LocalParameterization?>args[1])._local_parameterization,
                (<LocalParameterization?>args[2])._local_parameterization)

        elif len(args)==4:
            self._local_parameterization = new ceres.ProductParameterization(
                (<LocalParameterization?>args[0])._local_parameterization,
                (<LocalParameterization?>args[1])._local_parameterization,
                (<LocalParameterization?>args[2])._local_parameterization,
                (<LocalParameterization?>args[3])._local_parameterization)
        else:
            raise RuntimeError("number of sub parameterization should within 2~4.")

