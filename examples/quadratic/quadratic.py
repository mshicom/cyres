#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cyres import *
from cost_functions.wrappers import SimpleCostFunction

iter_cnt = [0]
def foo():
    iter_cnt[0] += 1
    print iter_cnt[0]

x = np.array([105.])
problem = Problem()
problem.add_residual_block(SimpleCostFunction(), SquaredLoss(), [x])

options = SolverOptions()
options.max_num_iterations = 50
options.linear_solver_type = LinearSolverType.DENSE_QR
options.trust_region_strategy_type = TrustRegionStrategyType.DOGLEG
options.dogleg_type = DoglegType.SUBSPACE_DOGLEG
options.minimizer_progress_to_stdout = False
options.add_callback(SimpleCallback(foo))
summary = Summary()

solve(options, problem, summary)
print summary.briefReport()
print summary.fullReport()
print x
