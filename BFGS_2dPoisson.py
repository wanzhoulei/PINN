'''
This file performs the BFGS on the 2d Poisson Equation with PINN
It uses random seed 0 and number of iterations = 1000
It solves the outputs in the results folder

'''

import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from PINN2D_tool import *
import time
import scipy.optimize

##change the number of iterations, damping coeff alpha and random seed here if needed
n_iter = 100; gamma = 1.99
write_pickle = True ##whether to serialize the PINN object
read_picke = None #provide the link to the .dat file to resume the GD
random_seed = 0 #set the random seed for generating the initial parameters of PINN

N = 50 ##use 40 by 40 grid points
layers = np.array([2, 20, 30, 20, 1]) #2 hidden layers
X_f_train, X_u_train, u_train = gridData(N)

maxcor = 200 ##maximum memory in L-BFGS 
max_iter = 10000

PINN = 0
if read_picke is not None:
    with open(read_picke, 'rb') as f:
        PINN = pickle.load(f)
else:
    PINN = Sequentialmodel(layers, seed=random_seed, N=N, gamma=gamma)
init_params = PINN.get_weights().numpy()

# train the model with Scipy newton-cg optimizer to run L2 GD
s = time.time()
if read_picke is None:
    PINN.start_time = s
else:
    PINN.start_time = s - PINN.clock[-1]
# train the model with Scipy L-BFGS optimizer
results = scipy.optimize.minimize(fun = PINN.optimizerfunc, 
                                  x0 = init_params, 
                                  args=(), 
                                  method='L-BFGS-B', 
                                  jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                  callback = PINN.optimizer_callback_lbfg, 
                                  options = {'disp': None,
                                            'maxcor': maxcor, 
                                            'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
                                            'gtol': 5e-10, 
                                            'maxfun':  50000, 
                                            'maxiter': n_iter,
                                            'iprint': -1,   #print update every 50 iterations
                                            'maxls': 50})

e = time.time()
print("Entire time: {}".format(e-s))
print("CPU time each iteration: {}".format((e-s)/n_iter))

#build the result folder if not exists
mydir = ("results")
if not os.path.isdir(mydir):
    os.makedirs(mydir)
##save the resulting convergence trace
tracepath = "./results/2dpoisson_BFGS_{}iter_seed{}_f_{}_N{}_{}-gamma{}.csv".format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

##save the plot
PINN.set_weights(results.x)
u_pred = PINN.evaluate(X_u_test)
u_pred = np.reshape(u_pred,(256,256),order='F') 
figpath = './results/2dpoisson_BFGS_{}iter_seed{}_f_{}_N{}_{}_gamma{}_solution.png'.format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
solutionplot(u_pred, usol, figpath)
plt.clf()
plt.cla()

##save the convergence trace plot
plt.figure(figsize=(8, 6))
pltpath = "./results/2dpoisson_BFGS_{}iter_seed{}_f_{}_N{}_{}_gamma{}_loss.png".format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
plt.plot(PINN.loss_trace)
plt.xlabel("number of iteration")
plt.ylabel("loss function")
plt.title("BFGS on 2d Poisson with PINN loss trace")
plt.savefig(pltpath, dpi=300)

if write_pickle:
    pickle_writepath = './results/2dpoisson_lbfgs_{}iter_seed{}_f_{}_N{}_{}_PINN_gamma{}.dat'.format(len(PINN.loss_trace), 
        random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
    with open(pickle_writepath, 'wb') as f:
        pickle.dump(PINN, f)