'''
This file performs the std Gradient Descent on the 2d Poisson Equation with PINN
It uses random seed 0 and number of iterations = 10000
It solves the outputs in the results folder

'''

import os
import numpy as np
from matplotlib import pyplot as plt
from PINN2D_tool import *
import time
import scipy.optimize

##change the number of iterations, damping coeff alpha and random seed here if needed
n_iter = 10000
random_seed = 0 #set the random seed for generating the initial parameters of PINN

N = 50 ##use N by N grid points
layers = np.array([2, 30, 30, 1]) #2 hidden layers
X_f_train, X_u_train, u_train = gridData(N)

PINN = Sequentialmodel(layers, seed=random_seed, N=N)
init_params = PINN.get_weights().numpy()

# train the model with Scipy newton-cg optimizer to run std GD
s = time.time()
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
print("Entire time: {}".format(e-s))
print("CPU time each iteration: {}".format((e-s)/n_iter))

#build the result folder if not exists
mydir = ("results")
if not os.path.isdir(mydir):
    os.makedirs(mydir)
##save the resulting convergence trace
tracepath = "./results/2dpoisson_GD_{}iter_seed{}_f_{}_N40_{}.csv".format(n_iter, 
    random_seed, str(func_RHS).split(' ')[1], layertostr(layers))
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

##save the plot
PINN.set_weights(results.x)
u_pred = PINN.evaluate(X_u_test)
u_pred = np.reshape(u_pred,(256,256),order='F') 
figpath = './results/2dpoisson_GD_{}iter_seed{}_f_{}_N40_{}_solution.png'.format(n_iter, 
    random_seed, str(func_RHS).split(' ')[1], layertostr(layers))
solutionplot(u_pred, usol, figpath)
plt.show()
plt.cla()


plt.plot(PINN.loss_trace)
plt.xlabel("number of iteration")
plt.ylabel("loss function")
plt.title("Std GD on 2d Poisson with PINN loss trace")
plt.show()
##save the convergence trace plot
pltpath = "./results/2dpoisson_GD_{}iter_seed{}_f_{}_N40_{}_loss.png".format(n_iter, 
    random_seed, str(func_RHS).split(' ')[1], layertostr(layers))
plt.savefig(pltpath, dpi=300)