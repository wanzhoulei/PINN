import tensorflow as tf
import datetime, os
#hide tf logs 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} 
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import time
from pyDOE import lhs         #Latin Hypercube Sampling

# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

# DATA PREPARATION ===================
x = np.linspace(-np.pi,np.pi,100)
u = 0 #solution
for k in range(1,6):
    u += np.sin(2*k*x)/(2*k)

# TEST DATA ===========================
X_u_test = tf.reshape(x,[100,1]) ##line space of -pi to pi
# Domain bounds
lb = X_u_test[0]  # [-pi]
ub = X_u_test[-1] # [pi]
u = tf.reshape(u,(100,1)) ##truth solution 

# TRAINING DATA =========================
##N_f is the number of collocation points
def trainingdata(N_f, sample=True):

    '''Boundary Conditions'''

    #Left egde (x = -π and u = 0)
    leftedge_x = -np.pi
    leftedge_u = 0
    
    #Right egde (x = -π and u = 0)
    rightedge_x = np.pi
    rightedge_u = 0

    X_u_train = np.vstack([leftedge_x, rightedge_x]) # X_u_train [2,1]
    u_train = np.vstack([leftedge_u, rightedge_u])   #corresponding u [2x1]

    '''Collocation Points'''

    if sample:
        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        X_f_train = lb + (ub-lb)*lhs(1,N_f) 
        X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points
    else:
        step = 2*np.pi/(N_f+1)
        X_f_train = np.arange(-np.pi, np.pi+step, step).reshape(-1, 1)

    #X_f_train is the collocation points and the boundary points
    #x_u_train is the 2 data point at endpoints
    #u_train is the truth value of the PDE at boundary points
    return X_f_train, X_u_train, u_train 

N_f = 1000 #Total number of collocation points 
# Training data
X_f_train, X_u_train, u_train = trainingdata(N_f, sample=False)
layers = np.array([1,20, 20,1]) #2 hidden layers

# PINN CLASS ===============================
class Sequentialmodel(tf.Module): 
    def __init__(self, layers, name=None, seed=10):
       
        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        self.loss_trace = []
        self.seed = seed
        self.layers = layers
        
        gen = tf.random.Generator.from_seed(seed=self.seed)
        for i in range(len(layers)-1):
            
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            #Xavier standard deviation 
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))

            #weights = normal distribution * Xavier standard deviation + 0
            w = gen.normal([input_dim, output_dim], dtype = 'float64') * std_dv
                       
            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = 'float64'), trainable = True, name = 'b' + str(i+1))
                    
            self.W.append(w)
            self.W.append(b)
            
            self.parameters +=  input_dim * output_dim + output_dim
    
    def evaluate(self,x):
        ##normalize x so that x is in [0, 1]
        x = (x-lb)/(ub-lb)
        
        a = x
        for i in range(len(self.layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.tanh(z)
            
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1]) # For regression, no activation to last layer
        return a
    
    #get the flattened weights and bias
    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(self.layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        
        return parameters_1d
        
    #parameters is falttened weights and bias
    def set_weights(self,parameters):
                
        for i in range (len(self.layers)-1):

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

    
    ##loss of the boundary points 
    ##x should be the two boundary points
    ##y should be the value of the PDE at these two boundary points
    def loss_BC(self,x,y):
        ##compute the MSE
        loss_u = tf.reduce_mean(tf.square(y-self.evaluate(x)))
        return loss_u

    ##compute the loss of collocation points
    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype = 'float64', trainable = False)
    
        nu = 0.01/np.pi

        x_f = g[:,0:1]

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x_f)

            z = self.evaluate(x_f)
            u_x = tape.gradient(z,x_f)

        u_xx = tape.gradient(u_x, x_f)

        del tape
        
        source = 0
        
        for k in range(1,6):
            source += (2*k) * np.sin(2*k*x_f) 
            
        ##f = u_xx + u
        f = u_xx + source

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f
    
    def loss(self,x,y,g, record = True):

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(g)

        loss = loss_u + loss_f
        if record:
            self.loss_trace.append(float(loss))

        return loss, loss_u, loss_f
    
    ##set the nn to the new parameters
    ##and compute and return the gradient of the nn w.r.t. parameters
    def optimizerfunc(self,parameters):
        
        self.set_weights(parameters)
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss_val, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train, record=False)
            
        grads = tape.gradient(loss_val,self.trainable_variables)
                
        del tape
        
        grads_1d = [ ] #flatten grads 
        
        for i in range (len(self.layers)-1):

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


## Gradient Descent algorithms ======================
def gradient_descent(PINN, initial, n_iter, lr, display_every=1):
    loss, gradient = PINN.optimizerfunc(initial)
    loss_trace = [loss]
    for i in range(n_iter):
        if (i+1)%display_every == 0:
            print("In {}th iteration. Current loss: {}".format(i+1, loss))
        new_params = PINN.get_weights().numpy() - lr*gradient
        loss, gradient = PINN.optimizerfunc(new_params)
        loss_trace.append(loss)
    return loss_trace

##this function plots the loss trace of GD as well as the output of the trained NN
def plot_nn(loss, PINN, lr=None, figpath=None):
    u_pred = PINN.evaluate(X_u_test)
    print("GD of lr={}, n_iter={}".format(lr, len(loss)))
    print("Train Loss: {}".format(loss[-1]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(loss)
    ax[0].set_xlabel("Number of Iterations")
    ax[0].set_ylabel("Train Loss")
    if lr is not None:
        ax[0].set_title("Loss Trace of {} iterations, lr={}".format(len(loss), lr))
    else:
        ax[0].set_title("Loss Trace of {} iterations".format(len(loss)))
    ax[1].plot(X_u_test, u_pred, label="PINN Solution")
    ax[1].plot(x,u, label="Truth Solution")
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u')
    ax[1].set_title("The Truth Solution and PINN Solution, train loss: {}".format(loss[-1]))
    ax[1].legend()
    if figpath is not None:
        plt.savefig(figpath, dpi=150)

##Gauss Newton Algorithms ===================================
#this function returns the gradient of the loss function  and the jacobian of the nn
#X_f_train, colocation points and the boundary points
#X_u_train 2 boundary points
#u_train values at boundary according to the boundary condition
#it returns (Jacobian, Gradient, loss_value)
def JacGrad(PINN, X_u_train, u_train, X_f_train):
    with tf.GradientTape(persistent=True) as tape:
        prediction = PINN.evaluate(X_f_train)
        loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train)
    Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
    grad = tape.gradient(loss_val, PINN.trainable_variables)
    return (Jac, grad, loss_val)

#N is the number of data points
def update_step(PINN, Jac, grad, N, lr):
    #flatten the gradient and construct the jacobian matrix
    numLayer = int(len(grad)/2) + 1
    grads_1d = [ ] #flatten grads     
    for i in range (numLayer -1):
        grads_w_1d = tf.reshape(grad[2*i],[-1]) #flatten weights 
        grads_b_1d = tf.reshape(grad[2*i+1],[-1]) #flatten biases
        grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
        grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
    Jacobian = tf.concat([tf.reshape(Jac[i],[1002, -1]) for i in range(len(Jac))],axis=1)
    numVars = Jacobian.shape[1]
    #compute the descent direction
    In = np.eye(numVars)
    B =  0.1*In + tf.transpose(Jacobian)@Jacobian
    st = np.linalg.solve(B, -grads_1d)
    #do update
    new_params = PINN.get_weights().numpy() + lr*st
    PINN.set_weights(new_params)

#N: number of data points, lr: learning rate, n_iter: number of iterations
def GaussNewton(PINN, X_u_train, u_train, X_f_train, N, lr, n_iter, display=10):
    loss_trace = []
    for i in range(n_iter):
        Jac, grad, loss_val = JacGrad(PINN, X_u_train, u_train, X_f_train)
        loss_trace.append(float(loss_val))
        if (i+1)%display==0:
            print("In {}th iteration, current loss: {}".format(i+1, loss_val))
        update_step(PINN, Jac, grad, N, lr)
    loss_trace.append(float(PINN.loss(X_u_train, u_train, X_f_train)[0]))
    return loss_trace

def Identity(x):
    return np.eye(481)

def L2Kernel(PINN, alpha=1):
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[1002, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        B =  alpha*In + tf.transpose(Jacobian)@Jacobian
        return B
    return func

## Kernel Functions for H1, H-1, H1 semi, H-1 semi
def Cn(N):
    ones = np.ones(N);
    diags = np.array([-1, 1]);
    data = [-ones, ones];
    C = sp.spdiags(data, diags, N, N)
    dx = 2*np.pi/(N+1)
    C = (1/(2*dx))*C
    return C.toarray()

# create a matrix of N+2 -by- N+2 , e.g., L
def NegLaplacian(N):
    C = Cn(N)
    L = np.eye(N+2)
    L[1:-1, 1:-1] = C.T @ C
    return L # without boundary points, it is N-by-N

def H1Kernel(PINN, N, X_f_train, L, alpha=0):
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[N+2, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(N+2)
        B =  tf.transpose(Jacobian)@(In + L)@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def H1semiKernel(PINN, N, X_f_train, L, alpha=0):
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[N+2, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@L@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def HinvKernel(PINN, N, X_f_train, L, alpha=0):
    In = np.eye(N+2)
    LIninv = np.linalg.inv(In + L)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[N+2, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@LIninv@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def HinvsemiKernel(PINN, N, X_f_train, L, alpha=0):
    Linv = np.linalg.inv(L)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[N+2, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@Linv@Jacobian
        return B + alpha*np.eye(numVars)
    return func

