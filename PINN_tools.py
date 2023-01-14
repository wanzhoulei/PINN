'''
This module provides all functionalities to solve a 1d poisson equation using PINN.

'''

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
    '''
    This method generates the discretized data points along the 1d segment [-pi, pi]. 
    It has two modes: either evenly distributed colocation data points
    or colocation data points generated by Latin Hypercube Sampling

    Parameters
    ----------
    N_f : int
        Number of colocation points 
    sample : bool
        If set to True, colocation points will be generated by Latin Hypercube Sampling
    
    Returns
    -------
    X_f_train : numpy.ndarray
        A 1d numpy array of shape (N_f+2,). It contains all N_f+2 data points along the 1d segement. 
        The two boundary points {-pi, pi} are placed at two ends.
        If sample=False, all colocation points will be ordered from left to right
    X_u_train : numpy.ndarray
        A 1d numpy array of shape (2,). In this case it is np.array([-pi, pi]), which is the set of boundary points.
    u_train : numpy.ndarray
        A 1d numpy array of shape (2,). The boundary condition on the two boundary points. 
        In this case it is np.array([0, 0])
    
    '''

    #Left egde (x = -π and u = 0)
    leftedge_x = -np.pi
    leftedge_u = 0
    
    #Right egde (x = -π and u = 0)
    rightedge_x = np.pi
    rightedge_u = 0

    X_u_train = np.vstack([leftedge_x, rightedge_x]) # X_u_train [2,1]
    u_train = np.vstack([leftedge_u, rightedge_u])   #corresponding u [2x1]

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


# PINN CLASS ===============================
class Sequentialmodel(tf.Module): 
    '''
    This class models the PINN neural nework. It extends the tf.Module class

    ...
    Attributes
    ----------
    W : list
        A list of tf.tensorflow objects that store the weights and biases of the PINN each each layer
    parameters : int
        Number of parameters in the PINN. Weights and biases. 
    loss_trace : list
        A list of floats that keep records of the total loss of the PINN during optimization iterations.
    seed : int
        The random seed used to generate the initial parameter values.
    layers : numpy.ndarray
        The shape of the PINN. For example, if the PINN has 2 input nodes and 2 hidden layers with 30 nodes and 1 output node, 
        then layers should be np.array([1, 30, 30, 1]).
    N : int
        The number of data points in each dimension in the frame.
    X_f_train : numpy.ndarray
        The array of all N**2 data points including BC points and colocation points. 
    X_u_train : numpy.ndarray
        The array of all boundary data points. 
    
    Methods
    -------
    evaluate(x)
        Input x to the PINN and returns the evaluation of the PINN on x
    get_weights()
        Get all parameters (weights and biases) as a 1d tf.tensor.
    set_weights(parameters)
        Set the parameters of the PINN (all weights and bias) according to intput parameters.
    loss_BC(x,y)
        Compute and return the loss_BC, the loss induced by the violation of the boundaty conditions.
    loss_PDE(x_to_train_f)
        Compute and return the loss_colocation. the loss induced by the colacation points.
    loss(x,y,g, record = False)
        Compute and return the total loss. Record the loss if record is true. 
    optimizerfunc(parameters)
        Set the parameters of the PINN by parameters. Compute and return the total loss and the gradient of loss function 
        w.r.t. all parameters as a 1d numpy.ndarray.
    optimizer_callback(parameters)
        Call back function for optimizers. 
    
    '''

    def __init__(self, layers, seed=10, N=1000):
        '''
        Constructor for Sequentialmodel
        All activation functions are tanh.
        It sets all the attributes, constructs and stores the training data points
        It also initialize the parameters of the PINN using the random seed input.
        It uses Xavier normal distribution to generate the initial parameters.
        
        Parameters
        ----------
        layers : numpy.ndarray
            1d numpy array that represents the shape of the PINN. For example, if the PINN has 
            3 hidden layers, each of which has 30 nodes, then layers should be np.array([2, 30, 30, 30, 1])
        seed : int, default=10
            The random seed used to initialize the parameters of PINN
        N : int, default=1000
            Number of colocation points
        
        '''
       
        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        self.loss_trace = []
        self.seed = seed
        self.layers = layers
        self.N = N
        X_f_train, X_u_train, u_train = trainingdata(N, sample=False)
        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train = u_train
        
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
        '''
        This method evaluates the PINN on the given input x.
        It imposes the neural network on each element in the input array x.

        Parameters
        ----------
        x : numpy.ndarray
            1d numpy array. Each element is one data point on which we want to evaluate the PINN.
        
        a : tensorflow.tensor 
            A tensor of the same shape as x. Each element of a is the evaluation of the PINN on the 
            corresponding data point.
        
        '''

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
        '''
        This method gets and returns all the parameters of the PINN and returns as a tensorflow.tensor object

        Returns
        -------
        tensorflow.tensor
            1d tensorflow.tensor object that contains all the flattened trainable parameters, including the weights and biases. 
        
        '''

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array       
        for i in range (len(self.layers)-1):          
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases        
        return parameters_1d
        
    #parameters is falttened weights and bias
    def set_weights(self,parameters):
        '''
        This method sets the parameters of the PINN according to the input.

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy array of all the flattened trainable parameters of the PINN including weights and biases. 
            The order should be the same as in get_weights method, i.e. layer by layer and weights then biases.
        
        '''
                
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
        '''
        This methods computes and returns boundary condition loss loss_BC of the current PINN.
        Given the training set and the boundary condition. 
        loss_BC should be the MSE of the boundary values of the PINN and the truth boundary values.

        Parameters
        ----------
        x : numpy.ndarray
            1d numpy array of length 2. It should be the boundary points, in this case np.array([-pi, pi])
        y : numpy.ndarray
            1d numpy array of length 2. It should be the values of the truth function at boundary points. 
            In this case, should be np.array([0, 0])
        
        '''

        loss_u = tf.reduce_mean(tf.square(y-self.evaluate(x)))
        return loss_u

    ##compute the loss of collocation points
    def loss_PDE(self, x_to_train_f):
        '''
        Computes and returns the colocation loss loss_f of the current PINN given inputs.
        loss_f is the MSE of the Laplacian of the PINN and the true Laplacian of the truth function 
        evaluated at colocation points.

        Parameters
        ----------
        x_to_train_f : numpy.ndarray
            1d numpy array of all the training data points, including the boundary and colocation points.

        Returns
        -------
        loss_f : tensorflow.tensor
            Scalar tensor object that is the colocation loss. 
        
        '''

        g = tf.Variable(x_to_train_f, dtype = 'float64', trainable = False)  
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
        '''
        Computes and returns the total loss of the PINN, which is the sum of BC loss and colocation loss.
        total_loss = loss_BC + loss_f

        Parameters
        ----------
        x : numpy.ndarray
            1d numpy array of all boundary points. In this case, it is np.array([-pi, pi])
        y : numpy.ndarray
            1d numpy array of boundary condition. The value of truth function at boundary points.
            In this case, it is np.array([0, 0])
        g : numpy.ndarray
            1d numpy array of all data points.
        record : bool, default=True
            If set to True, the total loss will be appended to the self.loss_trace list.
            If set to False, the loss will not be recorded.
        
        Returns
        -------
        loss : tensorflow.tensor
            Scalor tensor, which is the total loss of PINN.
        loss_u : tensorflow.tensor
            Scalor tensor, which is the boundary loss of PINN.
        loss_f : tensorflow.tensor
            Scalor tensor, which is the colocation loss of PINN.
        
        '''

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(g)
        loss = loss_u + loss_f
        if record:
            self.loss_trace.append(float(loss))
        return loss, loss_u, loss_f
    
    ##set the nn to the new parameters
    ##and compute and return the gradient of the nn w.r.t. parameters
    def optimizerfunc(self,parameters):  
        '''
        Optimizer function that sets the new parameters and computes and returns
        the loss and the gradient of loss w.r.t. all parameters

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy array of all falattened tunable parameters of the PINN, including weights and biases
            The order of parameters is important and should be the same as in set_weights method

        Returns
        -------
        numpy.ndarray
            The numpy array of one element, which is the total loss of the PINN after setting new parameters
        numpy.ndarray
            A 1d numpy array of the gradient of the loss function w.r.t. all parameters of PINN.
        
        '''   

        self.set_weights(parameters)    
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)       
            loss_val, loss_u, loss_f = self.loss(self.X_u_train, self.u_train, self.X_f_train, record=False)
            
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
        '''
        Optimizer callback function to be called after each optimization iteration
        It computes the losses (total loss, loss_BC, loss_colocation)
        and prints the number of iteration and these losses to the screen

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy array of all parameters of PINN

        '''         

        loss_value, loss_u, loss_f = self.loss(self.X_u_train, self.u_train, self.X_f_train, record=True)     
        tf.print("{}th iteration: total loss: {}, BC loss: {}, Colo loss: {}".format(
            len(self.loss_trace), loss_value, loss_u, loss_f
        ))



## Gradient Descent algorithms ======================
def gradient_descent(PINN, initial, n_iter, lr, display_every=1):
    '''
    Method that performs n_iter iterations of gradient descent on the PINN given initial parameter states

    Parameters
    ----------
    PINN : Sequentialmodel
        A Sequentialmodel object of the PINN we want to do GD on.
    initial : numpy.ndarray
        1d numpy array of the initial parameters of PINN
    n_iter : int
        Number of iterations to perform
    lr : float
        step size of the gradient descent 
    display_every : int, default=1
        After every display_every iterations, the loss value will be printed

    Returns
    -------
    loss_trace : list
        A list of loss values of each GD iteration.
    
    '''

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
    '''
    Plotting method that plots two subplots in one plot.
    1. The loss trace of some optimization algorithm
    2. The truth value of function and the PINN prediction in one subplot.

    Parameters
    ----------
    loss : array-like
        A 1d array of loss values through the optimization algorithm
    PINN : Sequentialmodel
        The PINN model we did the optimization on.
    lr : float, default=None
        The step size of the optimization algorithm
    figpath : String, default=None
        If set to None, the plot will not be saved.
        If you wish to save the plot, set it to be the saving path.
    
    '''

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
    '''
    This methods computes and returns the Jacobian, Gradient, and total loss
    Given all discretized training data points.
    Let g be a vector function that maps all parameters of PINN to the evaluation of the PINN on all data points.
    Suppose there are p parameters and N^2 data points, then g: R^p -> R^{N^2}
    The Jacobian is the Jacobian of g w.r.t. all parameters and has shape (N^2, p)
    Gradient is the gradient of loss w.r.t. all parameters

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model to compute the jacobian on
    X_u_train : numpy.ndarray
        1d numpy array of all boundary points
    u_train : numpy.ndarray
        1d numpy array of the evaluation of truth function on boundary points
    X_f_train : numpy.ndarray
        1d numpy array of all discretized data points

    Returns 
    -------
    Tuple 
        A tuple (Jac, grad, loss_val) is returned
        Jac is the tf.tensor of the Jacobian of PINN w.r.t. all parameters
        grad is the tf.tensor of the gradient of loss w.r.t. all parameters
        loss_val is the tf.tensor of the total loss of PINN
 
    '''

    with tf.GradientTape(persistent=True) as tape:
        prediction = PINN.evaluate(X_f_train)
        loss_val, _, _ = PINN.loss(X_u_train, u_train, X_f_train)
    Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
    grad = tape.gradient(loss_val, PINN.trainable_variables)
    return (Jac, grad, loss_val)

#N is the number of data points
def update_step(PINN, Jac, grad, lr):
    '''
    update step for Gauss Newton method
    It computes the descent direction for L2 natural gradient descent with 
    a damping factor of 0.1
    and then updates the descent step to the parameters

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model to perform the Gauss Newton on
    Jac : tensorflow.tensor
        The jacobian of the PINN w.r.t. all parameters. Its shape is complex and contains 
        one matrix for each weight and bias in every layer.
    grad L tensorflow.tensor
        The gradient of the loss w.r.t. all parameters. Its shape is complex. 
        It contains weight matrix and bias matrix and is not flattened yet
    lr : float
        The fixed step size of Gauss Newton method.
    
    '''

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
def GaussNewton(PINN, X_u_train, u_train, X_f_train, lr, n_iter, display=10):
    '''
    This method performs n_iter of fixed step size L2 natural gradient descent on the given PINN
    starting with the initial parameters of the current parameters of given PINN

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN to perform L2 NGD on
    X_u_train : numpy.ndarray
        The array of boundary points
    u_train : numpy.ndarray
        The numpy array of values of the truth function evaluated at the boundary points
    X_f_train : numpy.ndarray
        The numpy array of all the data points
    lr : float
        The fixed step size.
    n_iter : int
        The number of iterations to perform
    display : int, default=10
        Display the loss message every display iteration
    
    '''

    loss_trace = []
    for i in range(n_iter):
        Jac, grad, loss_val = JacGrad(PINN, X_u_train, u_train, X_f_train)
        loss_trace.append(float(loss_val))
        if (i+1)%display==0:
            print("In {}th iteration, current loss: {}".format(i+1, loss_val))
        update_step(PINN, Jac, grad, lr)
    loss_trace.append(float(PINN.loss(X_u_train, u_train, X_f_train)[0]))
    return loss_trace

def Identity(x):
    '''
    This method returns the identity matrix.
    This is the utility function that should be used as the Hess in the Newton-cg framework
    Such that standard gradient descent is performed

    Parameters
    ----------
    x : numpy.ndarray
        Should be the parameter arrays of the PINN with shape (numParameters, ).
        In practise, we only need the shape to match the number of parameters. 
    
    '''
    
    return np.eye(x.shape[0])

def L2Kernel(PINN, alpha=0):
    '''
    A decorator that returns a function that computes the L2 kernel of the PINN.
    Suppose the Jacobian of the neural network evaluated at each grid point w.r.t. all parameters is J (N**2, numVar).
    Then the L2 kernel is alpha*In + J.T J, where alpha is a damping factor.

    Parameters
    ----------
    PINN : Sequentialmodel 
        Sequantialmodel object. An instantiated PINN on which we want to compute the L2 kernel.
    alpha : float, default=0
        The damping factor we want to impose on the L2 kernel

    Returns
    -------
    func : function 
        A function that takes one required input and returns the L2 kernel of the PINN specified. With the data points and 
        damping factor specified in the decorator. The content of the input X is not important. The returned function 
        is only supposed to be used in the newton-cg framework to compute the L2 NGD.
    
    '''

    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(PINN.X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[1002, -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        B =  alpha*In + tf.transpose(Jacobian)@Jacobian
        return B
    return func

## Kernel Functions for H1, H-1, H1 semi, H-1 semi
def Cn(N):
    '''
    Constructs and returns discretized divergence operator matrix C. 
    Suppose vector v is the evaluation of a 1d scalar function u at N colocation points.
    Then, Cv is the gradient of u w.r.t. x evaluated a N colocation points. 
    The gradient is computed using finite differentiation and it assumes zero boundary condition.

    Parameters
    ----------
    N : int 
        The number of colocation points in 1D.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N, N) that is the discretized divergence operator matrix.
    
    '''

    ones = np.ones(N)
    diags = np.array([-1, 1])
    data = [-ones, ones]
    C = sp.spdiags(data, diags, N, N)
    dx = 2*np.pi/(N+1)
    C = (1/(2*dx))*C
    return C.toarray()

# create a matrix of N+2 -by- N+2 , e.g., L
def NegLaplacian(N):
    '''
    Constructs and returns the discretized negative laplacian matrix L.
    L is of shape (N+2, N+2), where N+2 is the number of all data points.
    If vector v is the evaluation of a 1d scalar function u evaluated at all N+2 data points, 
    then Lv is the evaluation of negative laplacian of u at all data points: -u_xx

    Parameters
    ----------
    N : int
        The number of colocation points in 1D
    
    Returns 
    -------
    L : numpy.ndarray
        Array of shape (N+2, N+2) that is the matrix of negative laplacian operator
    
    '''

    C = Cn(N)
    L = np.eye(N+2)
    L[1:-1, 1:-1] = C.T @ C
    return L # without boundary points, it is N-by-N

def H1Kernel(PINN, N, X_f_train, L, alpha=0):
    '''
    A decorator that returns a function that computes the H1 kernel of the PINN.
    Suppose the Jacobian of the evaluation of PINN on all grid data points w.r.t. all parameters is J,
    then the H1 kernel is then alpha*In + J.T (In + L) J
    where alpha is a damping factor and L is the negative laplacian matrix. 
    More detailed description can be found in the documentation.ipynb file.

    Parameters
    ----------
    PINN : Sequentialmodel
        A PINN object on which to compute the H1 kernel
    N : int
        The number of colocation points in 1d. 
    X_f_train : numpy.ndarray
        1d numpy array of all grid data points including BC points and colocation points
    L : numpy.ndarray
        numpy matrix of shape (N+2, N+2) that is the negative divergence operator matrix.
    alpha : float, default=0
        The damping factor that adds to the kernel

    Returns
    -------
    func : function 
        A function that takes a required input, of which the content is not important, and returns the H2 kernel matrix.
        of the specified PINN and damping factor. It is only supposed to be used in the newton-cg optimization framework.

    '''

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

