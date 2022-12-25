'''
This file performs the H1 NGD Algorithm on the 2d Poisson Equation with PINN
It uses damping factor of 0, random seed 0 and number of iterations = 1000
It solves the outputs in the results folder

'''

import os
import numpy as np
from matplotlib import pyplot as plt
from PINN2D_tool import *
import time
import scipy.optimize

##change the number of iterations, damping coeff alpha and random seed here if needed
n_iter = 2000; alpha=0; gamma=1.99
random_seed = 0 #set the random seed for generating the initial parameters of PINN

N = 50 ##use 50 by 50 grid points
layers = np.array([2, 20, 30, 20, 1]) #3 hidden layers
X_f_train, X_u_train, u_train = gridData(N)

PINN = Sequentialmodel(layers, seed=random_seed, N=N, gamma=gamma)
init_params = PINN.get_weights().numpy()
kernel = H1Kernel(PINN, N, X_f_train, alpha=alpha)

# train the model with Scipy newton-cg optimizer to run L2 GD
s = time.time()
PINN.start_time = s
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
tracepath = "./results/2dpoisson_H1_{}iter_seed{}_alpha{}_f_{}_N{}_{}_gamma{}.csv".format(n_iter, 
    random_seed, str(alpha).replace('.','d'), str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

#save the cpu time
tracepath2 = "./results/2dpoisson_H1_{}iter_seed{}_alpha{}_f_{}_N{}_{}_gamma{}_CPUtime.csv".format(n_iter, 
    random_seed, str(alpha).replace('.','d'), str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
np.savetxt(tracepath2, np.array(PINN.clock), delimiter=",")

##save the plot
PINN.set_weights(results.x)
u_pred = PINN.evaluate(X_u_test)
u_pred = np.reshape(u_pred,(256,256),order='F') 
figpath = './results/2dpoisson_H1_{}iter_seed{}_alpha{}_f_{}_N{}_{}_solution_gamma{}.png'.format(n_iter, 
    random_seed, str(alpha).replace('.','d'), str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
solutionplot(u_pred, usol, figpath)
plt.clf()
plt.cla()

plt.figure(figsize=(8, 6))
plt.plot(PINN.loss_trace)
plt.xlabel("number of iteration")
plt.ylabel("loss function")
plt.title("H1 NGD on 2d Poisson with PINN loss trace")
##save the convergence trace plot
pltpath = "./results/2dpoisson_H1_{}iter_seed{}_alpha{}_f_{}_N{}_{}_loss_gamma{}.png".format(n_iter, 
    random_seed, str(alpha).replace('.','d'), str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
plt.savefig(pltpath, dpi=300)

s = time.time()
k = kernel(1)
e = time.time()
print(e-s)


