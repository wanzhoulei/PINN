

import tensorflow as tf
import os
#hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'},
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import pandas as pd
import seaborn as sns 
import codecs, json

# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

##data prep ===================================
x_1 = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
x_2 = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]

X, Y = np.meshgrid(x_1,x_2) 

X_u_test = np.hstack((X.flatten(order='F')[:,None], Y.flatten(order='F')[:,None]))

# Domain bounds
lb = np.array([-1, -1]) #lower bound
ub = np.array([1, 1])  #upper bound

a_1 = 1 
a_2 = 1

usol = np.sin(a_1 * np.pi * X) * np.sin(a_2 * np.pi * Y) #solution chosen for convinience  

u = usol.flatten('F')[:,None] 
max_iter = 5000
maxcor = 200

## Training data ==================================
def trainingdata(N_u,N_f):
    
    leftedge_x = np.hstack((X[:,0][:,None], Y[:,0][:,None]))
    leftedge_u = usol[:,0][:,None]
    
    rightedge_x = np.hstack((X[:,-1][:,None], Y[:,-1][:,None]))
    rightedge_u = usol[:,-1][:,None]
    
    topedge_x = np.hstack((X[0,:][:,None], Y[0,:][:,None]))
    topedge_u = usol[0,:][:,None]
    
    bottomedge_x = np.hstack((X[-1,:][:,None], Y[-1,:][:,None]))
    bottomedge_u = usol[-1,:][:,None]
    
    all_X_u_train = np.vstack([leftedge_x, rightedge_x, bottomedge_x, topedge_x])
    all_u_train = np.vstack([leftedge_u, rightedge_u, bottomedge_u, topedge_u])  
     
    #choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False) 
    
    X_u_train = all_X_u_train[idx[0:N_u], :] #choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx[0:N_u],:]      #choose corresponding u
    
    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_f = lb + (ub-lb)*lhs(2,N_f) 
    X_f_train = np.vstack((X_f, X_u_train)) # append training points to collocation points 
    
    return X_f_train, X_u_train, u_train 

def gridData(N):
    dx = 2/float(N-1)
    X_f_train = []
    for i in range(N):
        for j in range(N):
            X_f_train.append([-1 + dx*i, -1 + dx*j])
    X_f_train = np.array(X_f_train)
    X_u_train = np.array([point for point in X_f_train if (abs(abs(point[0])-1)<1e-4 or abs(abs(point[1])-1)<1e-4)])
    u_train = np.zeros((X_u_train.shape[0], 1))
    return X_f_train, X_u_train, u_train

N = 40
X_f_train, X_u_train, u_train = gridData(N)
layers = np.array([2, 20, 20, 1]) #2 hidden layers

    
def plotData(X_f_train, X_u_train):
    fig,ax = plt.subplots()

    plt.plot(X_u_train[:,0],X_u_train[:,1], '*', color = 'red', markersize = 5, label = 'Boundary collocation points= 400')
    plt.plot(X_f_train[:,0],X_f_train[:,1], 'o', markersize = 0.5, label = 'PDE collocation points = 10,000')

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Collocation points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('scaled')
    plt.show()

##PINN -=============================
class Sequentialmodel(tf.Module): 
    def __init__(self, layers, name=None, seed=0):

        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        self.loss_trace = []
        self.seed = seed

        gen = tf.random.Generator.from_seed(seed=self.seed)
        for i in range(len(layers)-1):

            input_dim = layers[i]
            output_dim = layers[i+1]

            #Xavier standard deviation 
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))

            #weights = normal distribution * Xavier standard deviation + 0
            w = gen.normal([input_dim, output_dim], dtype = 'float64') * std_dv
            
#             w = tf.cast(tf.ones([input_dim, output_dim]), dtype = 'float64')

            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = 'float64'), trainable = True, name = 'b' + str(i+1))

            self.W.append(w)
            self.W.append(b)

            self.parameters +=  input_dim * output_dim + output_dim
            
        self.X = np.zeros(self.parameters) #store iterates
        self.G = np.zeros(self.parameters) #store gradients
        self.store = np.zeros((max_iter,2)) #store computed values for plotting
        self.iter_counter = 0 # iteration counter for optimizer
    
    def evaluate(self,x):
        
        #preprocessing input 
        x = (x - lb)/(ub - lb) #feature scaling
        
        a = x
        
        for i in range(len(layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.tanh(z)
            
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1]) # For regression, no activation to last layer
        
        return a
    
    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        
        return parameters_1d
        
    def set_weights(self,parameters):
                
        for i in range (len(layers)-1):

            shape_w = tf.shape(self.W[2*i]).numpy() # shape of the weight tensor
            size_w = tf.size(self.W[2*i]).numpy() #size of the weight tensor 
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # shape of the bias tensor
            size_b = tf.size(self.W[2*i+1]).numpy() #size of the bias tensor 
                        
            pick_w = parameters[0:size_w] #pick the weights 
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) # assign  
            parameters = np.delete(parameters,np.arange(size_w),0) #delete 
            
            pick_b = parameters[0:size_b] #pick the biases 
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign 
            parameters = np.delete(parameters,np.arange(size_b),0) #delete 

            
    def loss_BC(self,x,y):

        loss_u = tf.reduce_mean(tf.square(y-self.evaluate(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):
    
        g = tf.Variable(x_to_train_f, dtype = 'float64', trainable = False)

        k = 1    

        x_1_f = g[:,0:1]
        x_2_f = g[:,1:2]

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x_1_f)
            tape.watch(x_2_f)

            g = tf.stack([x_1_f[:,0], x_2_f[:,0]], axis=1)

            u = self.evaluate(g)
            u_x_1 = tape.gradient(u,x_1_f)
            u_x_2 = tape.gradient(u,x_2_f)

        u_xx_1 = tape.gradient(u_x_1,x_1_f)
        u_xx_2 = tape.gradient(u_x_2,x_2_f)

        del tape

        q = -( (a_1*np.pi)**2 + (a_2*np.pi)**2 - k**2 ) * np.sin(a_1*np.pi*x_1_f) * np.sin(a_2*np.pi*x_2_f)

        f = u_xx_1 + u_xx_2 + k**2 * u - q #residual

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f, f
    
    def loss(self,x,y,g, record = False):

        loss_u = self.loss_BC(x,y)
        loss_f, f = self.loss_PDE(g)

        loss = loss_u + loss_f
        if record:
            self.loss_trace.append(float(loss))

        return loss, loss_u, loss_f 
    
    def optimizerfunc(self,parameters):
        
        self.set_weights(parameters)
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss_val, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)
            
        grads = tape.gradient(loss_val,self.trainable_variables)
                
        del tape
        
        grads_1d = [ ] #store 1d grads 
        
        for i in range (len(layers)-1):

            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        
        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self,parameters):
                
        loss_value, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train, record=True)
        
        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)
        
        tf.print(loss_value, loss_u, loss_f, error_vec)
    
    def optimizer_callback_lbfg(self,parameters):
                
        loss_value, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train, record=True)
        
        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)
        
        tf.print(loss_value, loss_u, loss_f, error_vec)
        
        self.LbfgsInvHessProduct(parameters)
        
    def LbfgsInvHessProduct(self,parameters):

        self.iter_counter += 1  #update iteration counter 

        x_k = parameters  

        self.X = np.vstack((x_k.T,self.X)) #stack latest value on top row

        _,g_k = self.optimizerfunc(parameters) #obtain grads and loss value

        self.G = np.vstack((g_k.T,self.G)) #stack latest grads on top row

        n_corrs = min(self.iter_counter, maxcor) #for iterations < maxcor, we will take all available updates
        
        sk = self.X = self.X[:n_corrs] #select top 'n_corrs' x values, with latest value on top by construction
        yk = self.G = self.G[:n_corrs] #select top 'n_corrs' gradient values, with latest value on top by construction 

        #linear operator B_k_inv    
        hess_inv = scipy.optimize.LbfgsInvHessProduct(sk,yk) #instantiate class

        p_k = - hess_inv.matvec(g_k) #p_k = -B_k_inv * g_k

        gkpk = np.dot(p_k,g_k) #term 1 in report

        norm_p_k_sq = (np.linalg.norm(p_k,ord=2))**2 # norm squared
               
        #store the values
        self.store[self.iter_counter-1] = [gkpk,norm_p_k_sq]

        def ls_function(x):
            val, _ = self.optimizerfunc(x)
            return val

        def ls_gradient(x):
            _, grad = self.optimizerfunc(x)
            return grad

        alpha, _, _, fnewval, _, _ = scipy.optimize.line_search(ls_function, ls_gradient, x_k, p_k, gfk = g_k, maxiter = 50, c1=1e-4, c2=0.9)
        
        
        """
        Class
        -------------

        class scipy.optimize.LbfgsInvHessProduct(*args, **kwargs)

        sk = array_like, shape=(n_corr, n)
        Array of n_corr most recent updates to the solution vector.

        yk = array_like, shape=(n_corr, n)
        Array of n_corr most recent updates to the gradient.

        Methods
        -------------

        __call__(self, x)  Call self as a function.

        adjoint(self)      Hermitian adjoint.

        dot(self, x)       Matrix-matrix or matrix-vector multiplication.

        matmat(self, X)    Matrix-matrix multiplication.

        matvec(self, x)    Matrix-vector multiplication.

        rmatmat(self, X)   Adjoint matrix-matrix multiplication.

        rmatvec(self, x)   Adjoint matrix-vector multiplication.

        todense(self)      Return a dense array representation of this operator.

        transpose(self)    Transpose this linear operator.

        """
        
        
    def adaptive_gradients(self):

        with tf.GradientTape() as tape:
            tape.watch(self.W)
            loss_val = self.loss(X_u_train, u_train, X_f_train)

        grads = tape.gradient(loss_val,self.W)

        del tape

        return loss_val, grads

##kernels
def Identity(x):
    return np.eye(x.shape[0])

def L2Kernel(PINN, X_f_train, alpha=1):
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        B =  alpha*In + tf.transpose(Jacobian)@Jacobian
        return B
    return func

##plotting methods
def solutionplot(u_pred, usol, savepath=None):
    #color map
    cmap = cm.get_cmap('jet')
    normalize = colors.Normalize(vmin=-1, vmax=1)
    
    #Ground truth
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x_1, x_2, usol, cmap=cmap, norm=normalize)
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Ground Truth $u(x_1,x_2)$', fontsize=15)

    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolor(x_1, x_2, u_pred, cmap=cmap, norm=normalize)
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Predicted $\hat u(x_1,x_2)$', fontsize=15)

    # Error
    plt.subplot(1, 3, 3)
    plt.pcolor(x_1, x_2, abs(u_pred - usol), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(r'Absolute error $ | \hat u(x_1,x_2) - u(x_1,x_2) | $', fontsize=15)
    plt.tight_layout()
    
    if savepath is not None:
        plt.savefig(savepath, dpi = 500, bbox_inches='tight')