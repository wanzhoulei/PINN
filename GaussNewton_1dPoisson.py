'''
This file performs the Gauss Newton Algorithm on the 1d Poisson Equation with PINN
It uses damping factor of 0, random seed 20 and number of iterations = 7000
It solves the outputs in the results folder

'''

import os
import numpy as np
from matplotlib import pyplot as plt
from PINN_tools import *
import time
import scipy.optimize

##change the number of iterations, damping coeff alpha and random seed here if needed
n_iter = 7000; alpha=0
random_seed = 20 #set the random seed for generating the initial parameters of PINN

N_f = 1000 #Total number of collocation points 
# Training data
X_f_train, X_u_train, u_train = trainingdata(N_f, sample=False)
layers = np.array([1,20, 20,1]) #2 hidden layers

PINN = Sequentialmodel(layers, seed=random_seed)
init_params = PINN.get_weights().numpy()

##construct the l2 kernel without damping
kernel = L2Kernel(PINN, alpha=alpha)

s = time.time()
results = scipy.optimize.minimize(fun = PINN.optimizerfunc, 
                                  x0 = init_params, 
                                  args=(), 
                                  method='Newton-CG', 
                                  jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                  hess = kernel,
                                  callback = PINN.optimizer_callback, 
                                  options = {'disp': None,
                                             'maxiter': n_iter, 
                                             'xtol': 1e-20
                                            })
e = time.time()
print("Entire time: {}".format(e-s))
print("CPU time each iteration: {}".format((e-s)/n_iter)) 

#build the result folder if not exists
mydir = ("results")
if not os.path.isdir(mydir):
    os.makedirs(mydir)
##save the resulting convergence trace
tracepath = "./results/GN_{}iter_seed{}_alpha{}.csv".format(n_iter, random_seed, alpha)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

##save the plot
figpath = './results/GN_{}iter_seed{}_alpha{}_plot.png'.format(n_iter, random_seed, alpha)
plot_nn(PINN.loss_trace, PINN, lr=None, figpath=figpath)
