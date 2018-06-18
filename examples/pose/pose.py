#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cyres import *
from cost_functions import *
import numpy as np
import matplotlib.pyplot as plt

from numpy import sin,cos,pi
def rotateX(roll):
    """rotate around x axis"""
    return np.array([[1, 0, 0, 0],
                     [0, cos(roll), -sin(roll), 0],
                     [0, sin(roll), cos(roll), 0],
                     [0, 0, 0, 1]],'d')
def rotateY(pitch):
    """rotate around y axis"""
    return np.array([[cos(pitch), 0, sin(pitch),  0],
                     [0, 1, 0, 0],
                     [-sin(pitch), 0, cos(pitch), 0],
                     [0, 0, 0, 1]],'d')

def rotateZ(yaw):
    """rotate around z axis"""
    return np.array([[cos(yaw), -sin(yaw), 0, 0],
                     [sin(yaw), cos(yaw), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]],'d')

def translate(x,y,z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]],'d')

rand_range = lambda a,b : (b - a) * np.random.rand() + a

def plotTrajectory(pose_matrix, name=''):
    pose3d = np.array([p[:3,3] for p in pose_matrix]).T
    plt.plot(pose3d[0], pose3d[1], label=name)

oTc = translate(0, -0.1, 0)#.dot(rotateZ(-pi/180.0*15))#.dot(rotateX(-pi/2)).dot(rotateY(pi/2))
cTo = np.linalg.inv(oTc)
pose_odo = np.eye(4)
pose_cam = pose_odo.dot(oTc)
trj_odo, trj_cam = [SE3.create(pose_odo)], [SE3.create(pose_cam)]
for i in range(100):
    movement_odo = translate(rand_range(0, 0.1), rand_range(0, 0.1), rand_range(0, 0.1)).dot(rotateZ(pi/180*rand_range(-10, 10)))
    movement_cam = cTo.dot(movement_odo).dot(oTc)
#    pose_odo = pose_odo.dot(movement_odo)
#    pose_cam = pose_cam.dot(movement_cam)
    trj_odo.append(SE3.create(movement_odo))
    trj_cam.append(SE3.create(movement_cam))

plotTrajectory([p.matrix() for p in trj_odo], 'odo')
plotTrajectory([p.matrix() for p in trj_cam], 'cam')


#%%
oTc_est = SE3.create(np.eye(4))

problem = Problem()
problem.add_parameter_block(oTc_est.data, 7, LocalParameterizationSE3())

for move_odo, move_cam in zip(trj_odo, trj_cam):
    problem.add_residual_block(AdjointMotionCost(move_odo, move_cam),
                               SquaredLoss(),  oTc_est.data)




#%%
oTc_est = SE3.create(np.eye(4))
scale_est = np.array([1],'d')

problem = Problem()
problem.add_parameter_block(oTc_est.data, 7, LocalParameterizationSE3())
problem.add_parameter_block(scale_est, 1)
problem.set_parameter_lower_bound(scale_est, 0, 1e-5)

for pose in trj_odo:
    problem.add_parameter_block(pose.data, 7, LocalParameterizationSE3())
    problem.set_parameter_block_constant(pose.data)

for pose in trj_cam:
    problem.add_parameter_block(pose.data, 7, LocalParameterizationSE3())
    problem.set_parameter_block_constant(pose.data)

for i in range(len(trj_odo)):
    problem.add_residual_block(SimilarityCost(), SquaredLoss(),
                               oTc_est.data, trj_odo[i].data, trj_cam[i].data, scale_est)
#%%
options = SolverOptions()
options.max_num_iterations = 50
options.linear_solver_type = LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout = True

summary = Summary()

solve(options, problem, summary)
print summary.briefReport()
print oTc_est.matrix()
#print scale_est
print oTc

