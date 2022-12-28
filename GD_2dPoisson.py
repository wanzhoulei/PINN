'''
This file performs the std Gradient Descent on the 2d Poisson Equation with PINN
It uses random seed 0 and number of iterations = 10000
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
n_iter = 10; gamma=1.99
write_pickle = True
read_picke = None #provide the link to the .dat file to resume the GD
random_seed = 0 #set the random seed for generating the initial parameters of PINN

N = 50 ##use N by N grid points
layers = np.array([2, 20, 30, 20, 1]) #3 hidden layers
X_f_train, X_u_train, u_train = gridData(N)

PINN = 0
if read_picke is not None:
    with open(read_picke, 'rb') as f:
        PINN = pickle.load(f)
else:
    PINN = Sequentialmodel(layers, seed=random_seed, N=N, gamma=gamma)
init_params = PINN.get_weights().numpy()

# train the model with Scipy newton-cg optimizer to run std GD
s = time.time()
if read_picke is None:
    PINN.start_time = s
else:
    PINN.start_time = s - PINN.clock[-1]
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
tracepath = "./results/2dpoisson_GD_{}iter_seed{}_f_{}_N{}_{}_gamma{}.csv".format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

#save the cpu time
tracepath2 = "./results/2dpoisson_GD_{}iter_seed{}_f_{}_N{}_{}_gamma{}_CPUtime.csv".format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
np.savetxt(tracepath2, np.array(PINN.clock), delimiter=",")

##save the plot
PINN.set_weights(results.x)
u_pred = PINN.evaluate(X_u_test)
u_pred = np.reshape(u_pred,(256,256),order='F') 
figpath = './results/2dpoisson_GD_{}iter_seed{}_f_{}_N{}_{}_solution_gamma{}.png'.format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
solutionplot(u_pred, usol, figpath)
plt.clf()
plt.cla()

plt.figure(figsize=(8, 6))
plt.plot(PINN.loss_trace)
plt.xlabel("number of iteration")
plt.ylabel("loss function")
plt.title("Std GD on 2d Poisson with PINN loss trace")
##save the convergence trace plot
pltpath = "./results/2dpoisson_GD_{}iter_seed{}_f_{}_N{}_{}_loss_gamma{}.png".format(len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
plt.savefig(pltpath, dpi=300)

if write_pickle:
    pickle_writepath = './results/2dpoisson_GD_{}iter_seed{}_f_{}_N{}_{}_loss_gamma{}.dat'.format(len(PINN.loss_trace), 
        random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma)
    with open(pickle_writepath, 'wb') as f:
        pickle.dump(PINN, f)