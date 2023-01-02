
import sys
sys.path.append("..")
from PINN2D_tool import *
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import time

n_iter = 2000; gamma=1.99; BC_ratio = 1; Interior_ratio=.5; lr=32; alpha=0.01
write_pickle = True
read_picke = None #provide the link to the .dat file to resume the GD
random_seed = 0 #set the random seed for generating the initial parameters of PINN

N = 80 ##use N by N grid points
layers = np.array([2, 20, 30, 20, 1]) #3 hidden layers
X_f_train, X_u_train, u_train = gridData(N)
##create the negative laplacian matrix
L = NegLaplacian(N)

PINN = 0
if read_picke is not None:
    with open(read_picke, 'rb') as f:
        PINN = pickle.load(f)
else:
    PINN = Sequentialmodel(layers, seed=random_seed, N=N, gamma=gamma)

s = time.time()
if read_picke is None:
    PINN.start_time = s
else:
    PINN.start_time = s - PINN.clock[-1]

##do H1 NGD
H1_optimizer(PINN, lr, n_iter, L, BC_ratio=BC_ratio, Interior_ratio=Interior_ratio, alpha=alpha)


e = time.time()
print("Entire time: {}".format(e-s))
print("CPU time each iteration: {}".format((e-s)/n_iter))

#build the result folder if not exists
mydir = ("results")
if not os.path.isdir(mydir):
    os.makedirs(mydir)
##save the resulting convergence trace
tracepath = "./results/2dpoisson_H1mini_lr{}_{}iter_seed{}_f_{}_N{}_{}_gamma{}_BCRatio{}_IntRatio{}_alpha{}.csv".format(lr, len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma, BC_ratio, Interior_ratio, alpha)
np.savetxt(tracepath, np.array(PINN.loss_trace), delimiter=",")

#save the cpu time
tracepath2 = "./results/2dpoisson_H1mini_lr{}_{}iter_seed{}_f_{}_N{}_{}_gamma{}_BCRatio{}_IntRatio{}__alpha{}_CPUtime.csv".format(lr, len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma, BC_ratio, Interior_ratio, alpha)
np.savetxt(tracepath2, np.array(PINN.clock), delimiter=",")

##save the plot
u_pred = PINN.evaluate(X_u_test)
u_pred = np.reshape(u_pred,(256,256),order='F') 
figpath = './results/2dpoisson_H1mini_lr{}_{}iter_seed{}_f_{}_N{}_{}_solution_gamma{}_BCRatio{}_IntRatio{}_alpha{}.png'.format(lr, len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma, BC_ratio, Interior_ratio, alpha)
solutionplot(u_pred, usol, figpath)
plt.clf()
plt.cla()

plt.figure(figsize=(8, 6))
plt.plot(PINN.loss_trace)
plt.xlabel("number of iteration")
plt.ylabel("loss function")
plt.title("Mini batch H1 NGD on 2d Poisson with PINN loss trace")
##save the convergence trace plot
pltpath = "./results/2dpoisson_H1mini_lr{}_{}iter_seed{}_f_{}_N{}_{}_loss_gamma{}_BCRatio{}_IntRatio{}_alpha{}.png".format(lr, len(PINN.loss_trace), 
    random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma, BC_ratio, Interior_ratio, alpha)
plt.savefig(pltpath, dpi=300)

if write_pickle:
    pickle_writepath = './results/2dpoisson_H1mini_lr{}_{}iter_seed{}_f_{}_N{}_{}_PINN_gamma{}_BCRatio{}_IntRatio{}_alpha{}.dat'.format(lr, len(PINN.loss_trace), 
        random_seed, str(func_RHS).split(' ')[1], N, layertostr(layers), gamma, BC_ratio, Interior_ratio, alpha)
    with open(pickle_writepath, 'wb') as f:
        pickle.dump(PINN, f)