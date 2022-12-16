'''
This file does the std Gradient Descent Algorithm on the PINN
using 80000 iterations and random seed 20 to initialize the PINN parameters
The outputs are saved in the results folder

'''


import os
import numpy as np
from matplotlib import pyplot as plt
from PINN_tools import *
import time
import scipy.optimize

##change the number of iterations or random seed here if needed
n_iter = 80000
random_seed = 20 #set the random seed for generating the initial parameters of PINN

N_f = 1000 #Total number of collocation points 
# Training data
X_f_train, X_u_train, u_train = trainingdata(N_f, sample=False)
layers = np.array([1,20, 20,1]) #2 hidden layers

PINN = Sequentialmodel(layers, seed=random_seed)
init_params = PINN.get_weights().numpy()

s = time.time()
# train the model with Scipy newton-cg optimizer to run std GD
results = scipy.optimize.minimize(fun = PINN.optimizerfunc, 
                                  x0 = init_params, 
                                  args=(), 
                                  method='Newton-CG', 
                                  jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                  hess = Identity,
                                  callback = PINN.optimizer_callback, 
                                  options = {'disp': None,
                                             'maxiter': n_iter, 
                                             'xtol': 1e-20
                                            })
e = time.time()
print("Total time of GD of {} iterations: {}".format(n_iter, e-s))
print("Average time of each GD iteration: {}".format((e-s)/n_iter))

#build the result folder if not exists
mydir = ("results")
if not os.path.isdir(mydir):
    os.makedirs(mydir)
##save the resulting convergence trace
tracepath = "./results/GD_{}iter_seed{}.csv".format(n_iter, random_seed)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

##save the plot
figpath = './results/GD_{}iter_seed{}_plot.png'.format(n_iter, random_seed)
plot_nn(PINN.loss_trace, PINN, lr=None, figpath=figpath)