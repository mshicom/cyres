#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cyres import *
from cost_functions import *
import numpy as np


def rotateX(roll):
    """rotate around x axis"""
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(roll), -np.sin(roll), 0],
                     [0, np.sin(roll), np.cos(roll), 0],
                     [0, 0, 0, 1]],'d')
def test(T_w_targ, T_w_init):
    T_wr = T_w_init
    problem = Problem()

    problem.add_parameter_block(T_wr.data, 7, LocalParameterizationSE3())
    problem.add_residual_block(TestCostFunctor(T_w_targ.inverse()),
                               SquaredLoss(),
                               T_wr.data)

    options = SolverOptions()
    options.gradient_tolerance = 0.01 * 1e-10
    options.function_tolerance = 0.01 * 1e-10
    options.linear_solver_type = LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True

    summary = Summary()
    solve(options, problem, summary)
    print summary.fullReport()
    mse = np.linalg.norm((T_w_targ.inverse() * T_wr).log())
    print mse
    return mse < (10. * 1e-10)

a_targ = SE3.create(rotateX(np.pi/180*5))
a_init = SE3.create(np.eye(4))
print test(a_targ, a_init)

