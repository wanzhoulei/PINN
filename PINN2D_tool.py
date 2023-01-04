'''
This module provides all functionalities to solve a 2d poisson equation using PINN.

'''

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
import time
from pyDOE import lhs         #Latin Hypercube Sampling
from optimizer.optimizer import *

# generates same random numbers each time
tf.random.set_seed(1234)

##data prep ===================================
##create 256 by 256 test data points in the frame [-1, 1]
x_1 = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
x_2 = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]

X, Y = np.meshgrid(x_1,x_2) 

#concatenate X, Y to form the test data set  
X_u_test = np.hstack((X.flatten(order='F')[:,None], Y.flatten(order='F')[:,None]))

# Domain bounds
lb = np.array([-1, -1]) #lower bound
ub = np.array([1, 1])  #upper bound

## u = sin(pi x)sin(pi y)
def u1(x, y):
    '''
    The function u1(x,y) = sin(pi x)sin(pi y)

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the function u1
    y : float
        The value in y direction to evaluate the function u1

    Returns
    -------
    float
        The value of function u1 at point (x, y)
    
    '''
    return np.sin( np.pi * x) * np.sin( np.pi * y)

##the divergence of u1, i.e. u1_xx + u1_yy
def div1(x, y):
    '''
    The divergence of function u1(x,y) = sin(pi x)sin(pi y).
    u1_xx + u1_xx evaluated at (x, y)

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the divergence of function u1
    y : float
        The value in y direction to evaluate the divergence of function u1

    Returns
    -------
    float
        The divergence of function u1 at point (x, y)
    
    '''

    return -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)

##u = sin(pi x)sin(pi y) + sin(6pi x)sin(6pi y)
def u2(x, y):
    '''
    The function u2(x,y) = sin(pi x)sin(pi y) + sin(6pi x)sin(6pi y)

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the function u2
    y : float
        The value in y direction to evaluate the function u2

    Returns
    -------
    float
        The value of function u2 at point (x, y)
    
    '''

    return np.sin( np.pi * x) * np.sin( np.pi * y) + np.sin( 6*np.pi * x) * np.sin( 6*np.pi * y)

##divergence of u2
def div2(x, y):
    '''
    The divergence of function u2(x,y) = sin(pi x)sin(pi y) + sin(6pi x)sin(6pi y)
    u2_xx + u2_yy evaluated at point (x, y).

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the divergence of function u2
    y : float
        The value in y direction to evaluate the divergence of function u2

    Returns
    -------
    float
        The divergence of function u2 at point (x, y)
    
    '''

    return -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y) - 72*np.pi**2*np.sin(6*np.pi*x)*np.sin(6*np.pi*y)

##u3 = sin(pi x)sin(pi y) + sin(3pi x)sin(3pi y)
def u3(x, y):
    '''
    The function u3(x,y) = sin(pi x)sin(pi y) + sin(3pi x)sin(3pi y)

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the function u3
    y : float
        The value in y direction to evaluate the function u3

    Returns
    -------
    float
        The value of function u3 at point (x, y)
    
    '''

    return np.sin( np.pi * x) * np.sin( np.pi * y) + np.sin( 3*np.pi * x) * np.sin( 3*np.pi * y)

def div3(x, y):
    '''
    The divergence of function u3(x,y) = sin(pi x)sin(pi y) + sin(3pi x)sin(3pi y)
    u3_xx + u3_yy evaluated at point (x, y)

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the divergence of function u3
    y : float
        The value in y direction to evaluate the diergence of function u3

    Returns
    -------
    float
        The divergence of function u3 at point (x, y)
    
    '''

    return -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y) - 18*np.pi**2*np.sin(3*np.pi*x)*np.sin(3*np.pi*y)

##add constant 3 to u3
def u4(x, y):
    '''
    The function u4(x,y) = sin(pi x)sin(pi y) + sin(3pi x)sin(3pi y) + 3

    Parameters
    ----------
    x : float
        The value in x direction to evaluate the function u4
    y : float
        The value in y direction to evaluate the function u4

    Returns
    -------
    float
        The value of function u4 at point (x, y)
    
    '''

    return np.sin( np.pi * x) * np.sin( np.pi * y) + np.sin( 3*np.pi * x) * np.sin( 3*np.pi * y) + 3

##change these two lines if you want to use other functions
func_RHS = u4 ##set this variable to the function of true solution
div = div3 ##set this variable to be the divergence of the above function 
min_val = 1; max_val=5 ##set the minimum and maximum values of the function 

#generate the truth solution
usol = func_RHS(X, Y) #solution chosen for convinience  
u = usol.flatten('F')[:,None] 
##set hyperparameters for lbfgs
max_iter = 200000
maxcor = 200

## Training data ==================================
def trainingdata(N_u,N_f):
    '''
    trainingdata method constructs the train set of 2d data points for the PINN model. 
    It evenly selects boundary points to form the boundary data set and use Latin Hypercube Sampling to generate colocation data points.

    Parameters
    ----------
    N_u : int
        The number of boundary condition points to include.
    N_f : int
        The number of colocation points to include.

    Returns
    -------
    X_f_train : numpy.ndarray
        2d numpy.ndarray of shape (N_f+N_u, 2). It contains all generated boundary condition points and colocation points.
        BC points come after the colocation points.
    X_u_train : numpy.ndarray
        2d numpy.ndarray of shape (N_u, 2). It contains all generated boudary points.
    u_train : numpy.ndarray
        1d numpy.ndarray of shape (N_u,). It is the booundary condition and contains all truth value of the function at boundary.

    '''
    
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
    
    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_f = lb + (ub-lb)*lhs(2,N_f) 
    X_f_train = np.vstack((X_f, X_u_train)) # append training points to collocation points 
    
    return X_f_train, X_u_train, u_train 

def gridData(N, boundary=0):
    '''
    gridData method creates and returns evenly distributed gridpoints in the frame [-1, 1]*[-1, 1] as the training data.

    Parameters
    ----------
    N : int
        The number of gridpoints in each dimension. For example, N=40 means to draw 40 by 40 points in the frame.
    boundary : float, optional, default=0
        The boundary condition. The value of the truth function at the boundary points. 
        Default is to impose zero boundary condition.

    Returns
    -------
    X_f_train : numpy.ndarray
        2d numpy.ndarray of shape (N^2, 2). It contains all the N by N evenly distributed gridpoints in the frame. 
        The order the data points are aligned is important. Suppose the N by N data points of the frame form a matrix in 2d. 
        This method append data points columnwise. i.e. it gos through the data points matrix column by column from lower to 
        upper and from left to right. 
    X_u_train : numpy.ndarray
        2d numpy.ndarray of shape (N^2 - (N-2)^2, 2). It contains all the boundary condition points in the frame. 
        The ordering is the same as described above but only contains the boundary points. 
    u_train : numpy.ndarray
        1d numpy.ndarray of shape (N^2 - (N-2)^2,). It contains the evaluation of the truth function at the boundary points.
    
    '''
    dx = 2/float(N-1)
    X_f_train = []
    for i in range(N):
        for j in range(N):
            X_f_train.append([-1 + dx*i, -1 + dx*j])
    X_f_train = np.array(X_f_train)
    X_u_train = np.array([point for point in X_f_train if (abs(abs(point[0])-1)<1e-4 or abs(abs(point[1])-1)<1e-4)])
    u_train = np.zeros((X_u_train.shape[0], 1))+boundary
    return X_f_train, X_u_train, u_train

    
def plotData(X_f_train, X_u_train):
    '''
    Utility plotting function to show all the train data points.

    Parameters
    ----------
    X_f_train : array-like
        The array of all training points
    X_u_train : array-like
        The array of all boundary points
    
    '''

    fig,ax = plt.subplots()
    plt.plot(X_u_train[:,0],X_u_train[:,1], '*', color = 'red', markersize = 5, label = 'Boundary points= {}'.format(X_u_train.shape[0]))
    plt.plot(X_f_train[:,0],X_f_train[:,1], 'o', markersize = 0.5, label = 'PDE collocation points = {}'.format(X_f_train.shape[0]-X_u_train.shape[0]))

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Collocation points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('scaled')
    plt.show()

##PINN -=============================
#gamma is the weight for the boundary points
# loss = gamma*loss_BC + (1-gamma)*loss_colocation
class Sequentialmodel(tf.Module): 
    '''
    This class models the PINN neural network. It extends the tf.Module class
    
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
        then layers should be np.array([2, 30, 30, 1]).
    N : int
        The number of data points in each dimension in the frame.
    X_f_train : numpy.ndarray
        The array of all N**2 data points including BC points and colocation points. 
    X_u_train : numpy.ndarray
        The array of all boundary data points. 
    u_train : numpy.ndarray
        The array of the evaluation of the truth function at boundary points.
    X_interior : numpy.ndarray
        The array of the colocation points
    gamma : float
        The weight assigned to the boundary condition loss. The total loss is: loss = gamma*loss_BC + (1-gamma)*loss_colocation
    start_time : float
        The time when the optimization algorithm starts on the PINN
    clock : list
        A list of floats that keep records of the relative time of each iteration of optimization w.r.t. the start time
        i.e. time(iteration) - start_time
    boundary : float
        The boundary condition. The value of the truth function at boundary points. 
    X : numpy.ndarray
        Stores the iterates only for lbfgs optimization. Initialized to all zero.
    G : numpy.ndarray
        Stores the gradients only for lbfgs optimization. Initialized to all zeros.
    store : numpy.ndarray
        Store computed values for plotting, only for lbfgs optimization.
    iter_counter : int
        Counter for the number of iterations, only for lbfgs optimization. 

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
    optimizer_callback_lbfg(parameters)
        Call back function for lbfgs optimizer
    LbfgsInvHessProduct(parameters)
        Compute and store the inverse Hessian product, only for lbfgs optimizer. 
    
    '''

    def __init__(self, layers, seed=0, N=40, gamma=1, boundary=0):
        '''
        Constructor for the Sequentialmodel class. It sets the initial values of all attributes. 
        It initializes the parameters values of PINN using the random seed. It utilizes the Xavier normal distribution. 

        Parameters
        ----------
        layers : numpy.ndarray
            The shape of the neural network
        seed : int, default=0
            The random seed used to initilaize the parameters
        N : int, default=40
            The number of data points in each dimension in the frame
        gamma : float, default=1
            The weight imposed on the boundary consition loss
        boundary : float, default=0
            The boundary value of the truth function   

        '''

        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        self.loss_trace = []
        self.seed = seed
        self.layers = layers
        self.N = N
        X_f_train, X_u_train, u_train = gridData(N, boundary=boundary)
        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train = u_train
        self.X_interior = np.array([point for point in X_f_train if (abs(abs(point[0])-1)>=1e-4 and abs(abs(point[1])-1)>=1e-4)])
        self.gamma = gamma
        self.start_time = None
        self.clock = []
        self.boundary = boundary

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

        if self.boundary != 0:
                bound = tf.cast(self.boundary*tf.ones([layers[-1]]), dtype='float64')
                self.W[-1].assign_add(bound)
            
        self.X = np.zeros(self.parameters) #store iterates
        self.G = np.zeros(self.parameters) #store gradients
        self.store = np.zeros((max_iter,2)) #store computed values for plotting
        self.iter_counter = 0 # iteration counter for optimizer
    
    def evaluate(self,x):
        '''
        Evaluate the PINN on the points in x and returns the values as a numpy.ndarray

        Parameters
        ----------
        x : numpy.ndarray
            A numpy.ndarray of shape (n, 2), which contains n data points.
        
        Returns
        -------
        a : tf.tensor
            of shape (n,). It is the outputs of the PINN of the n inputs. 
        
        '''
        
        #preprocessing input 
        x = (x - lb)/(ub - lb) #feature scaling
        
        a = x
        
        for i in range(len(self.layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.tanh(z)
            
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1]) # For regression, no activation to last layer
        
        return a
    
    def get_weights(self):
        '''
        This methods returns all parameters of the PINN (weights and biases of each layer).
        These parameters are flattened into 1d array.

        Returns
        -------
        parameters_1d : tf.tensor
            1d tensor that stores all parameters values of the PINN (weights and biases of each layer)
        
        '''

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(self.layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        
        return parameters_1d
        
    def set_weights(self,parameters):
        '''
        This method sets the parameters of the PINN (weights and biases of every layer)

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy.ndarray that contains the values of the parameters of the PINN.

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

            
    def loss_BC(self,x,y):
        '''
        Computes and returns the MSE of the evaluation of the PINN on boundary points.

        Parameters
        ----------
        x : numpy.ndarray
            The array of the boundary condition points to evaluate at
        y : array-like
            The array of the boundary truth values at the corresponding boundary points

        Returns
        -------
        loss_u : tf.tensor
            The loss value. i.e. the MSE of the evaluation of the PINN on boundary points
        
        '''

        loss_u = tf.reduce_mean(tf.square(y-self.evaluate(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):
        '''
        Computes and returns the loss_colocation. Given the divergence function, it computes the MSE of the divergence 
        of the PINN on colocation points as the loss value.

        Parameters
        ----------
        x_to_train_f : numpy.ndarray
            The numpy.array that stores all the data points (including the BC points and the colocation points)
        
        Returns
        -------
        loss_f : tf.tensor
            The MSE of the divergence of the PINN
        f : tf.tensor
            The residual of the divergences of the PINN with the truth divergence
        
        '''
    
        g = tf.Variable(x_to_train_f, dtype = 'float64', trainable = False)

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

        #q = -( (a_1*np.pi)**2 + (a_2*np.pi)**2 - k**2 ) * np.sin(a_1*np.pi*x_1_f) * np.sin(a_2*np.pi*x_2_f)
        q = div(x_1_f, x_2_f)

        f = u_xx_1 + u_xx_2 - q #residual

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f, f
    
    def loss(self,x,y,g, record = False):
        '''
        Computes and returns the total loss, boundary condition loss and colocation loss of the PINN.
        Record the loss if the record flag is set to True. 

        Parameters
        ----------
        x : numpy.ndarray
            The numpy array of all the boundary points 
        y : numpy.ndarray
            The numpy array of the truth value of the function on the corresponding boundary points
        g : numpy.ndarray
            The numpy array of all the data points  (including the BC points and the colocation points)
        record : bool, default=False
            If set to True, the total loss will be stored in the loss_trace attribute

        Returns
        -------
        loss : tf.tensor
            The total loss. Which is a linear combination of the loss_BC and loss_colocation.
            loss = gamma*loss_BC + (2-gamma)*loss_colocation
        loss_u : tf.tensor
            The loss_colocation. The MSE of the divergences on the colocation points.
        loss_f : tf.tensor
            The loss_BC, the MSE of the evaluation of the PINN on boundary points.

        '''

        loss_u = self.loss_BC(x,y)
        loss_f, f = self.loss_PDE(g)

        loss = self.gamma*loss_u + (2-self.gamma)*loss_f
        if record:
            self.loss_trace.append(float(loss))

        return loss, loss_u, loss_f 
    
    def optimizerfunc(self,parameters):
        '''
        this method sets the parameters of the PINN according to the input parameters. 
        It returns the total loss and the gradient of the loss function w.r.t. the PINN parameters.

        Parameters
        ----------
        parameters : numpy.ndarray 
            1d numpy array that stores all the parameter values of the PINN (including the weights and biases of each layer)

        Returns
        -------
        numpy.ndarray
            The total loss of the PINN after updating the parameters. 
        numpy.ndarray 
            1d numpy array that stores the gradient of the loss function w.r.t. all parameters of PINN
        
        '''
        
        self.set_weights(parameters)
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss_val, loss_u, loss_f = self.loss(self.X_u_train, self.u_train, self.X_f_train)
            
        grads = tape.gradient(loss_val,self.trainable_variables)
                
        del tape
        
        grads_1d = [ ] #store 1d grads 
        
        for i in range (len(self.layers)-1):

            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        
        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self,parameters):
        '''
        callback function for various optimizers.
        It computes the total loss, loss_BC and loss_colocation. 
        Records the relative cpu time in the clock attribute. 
        Print out all loss.

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy.ndarray that stores the parameters of the PINN.
        
        '''

        loss_value, loss_u, loss_f = self.loss(self.X_u_train, self.u_train, self.X_f_train, record=True)
        
        self.clock.append(time.time()-self.start_time)
        
        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}'.format(len(self.loss_trace), loss_value, loss_f, loss_u))
    
    def optimizer_callback_lbfg(self,parameters):
        '''
        callback function only for lbfgs optimizer.
        It computes the total loss, loss_BC and loss_colocation. 
        Records the relative cpu time in the clock attribute. 
        Print out all loss.

        Parameters
        ----------
        parameters : numpy.ndarray
            1d numpy.ndarray that stores the parameters of the PINN.
        
        '''
                
        loss_value, loss_u, loss_f = self.loss(self.X_u_train, self.u_train, self.X_f_train, record=True)
        
        self.clock.append(time.time()-self.start_time)
        
        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}'.format(self.iter_counter, loss_value, loss_f, loss_u))
        
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
        

##kernels
def Identity(x):
    '''
    returns an identity matrix. It is used to do std Gradient Descent using the newton-cg framework.

    Parameters
    ----------
    x : numpy.ndarray
        Should be the parameter arrays of the PINN with shape (numParameters, ).
        In practise, we only need the shape to match the number of parameters. 

    '''
    return np.eye(x.shape[0])

def L2Kernel(PINN, X_f_train, alpha=0):
    '''
    A decorator that returns a function that computes the L2 kernel of the PINN.
    Suppose the Jacobian of the neural network evaluated at each grid point w.r.t. all parameters is J (N**2, numVar).
    Then the L2 kernel is alpha*In + J.T J, where alpha is a damping factor.

    Parameters
    ----------
    PINN : Sequentialmodel 
        Sequantialmodel object. An instantiated PINN on which we want to compute the L2 kernel.
    X_f_train : numpy.ndarray
        The (N**2, 2) numpy array of the entire grid data points including the BC points and colocation points. 
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
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        B =  alpha*In + tf.transpose(Jacobian)@Jacobian
        return B
    return func

#N is the number of grid points in one dim including the boundary points
#N-2 is the number of colocation grid points in one dim  
def Cn(N, dx, dy):
    '''
    This method computes and returns the divergence operator matrix C. (using finite differentiation)
    Suppose f is a 2d vector function. C acting on the output of f is equivalent to taking the y directional and then x directional derivative.
    Detailed description can be found in the documentation.ipynb file.

    Parameters
    ----------
    N : int
        The number of data points in one dimension. The number of entire data points is then N**2.
        The vector function we want to take the divergence of then has shape f: R^2 ---> R^{N**2}.
    dx : float
        The horizontal distance between each discretized grid points in the frame.
    dy : float
        The vertical distance between each discretized grid points in the frame.

    Returns
    -------
    numpy.ndarray
        A (2(N-2)^2, (N-2)^2) numpy matrix, that is the discretized divergence operator.
        Note that it is (N-2)^2 instead of N^2 because the matrix only acts on the evaluation of the function on interior points.
        It also assumes zero boundary condition. 

    '''

    ones = np.ones(N-2);
    diags = np.array([-1, 1]);
    data = [-ones, ones];
    C = sp.spdiags(data, diags, N-2, N-2).toarray();
    I = np.identity(N-2);
    A1 = sp.kron(I, C).toarray()/(2*dx);
    A2 = sp.kron(C, I).toarray()/(2*dy);
    return np.concatenate((A1, A2), axis=0);

def NegLaplacian(N):
    '''
    This matrix computes and returns the negative laplacian operator L for the PINN.
    For a 2d vector function, this matrix L acting on the output will turn each entry into negative laplacian. 
    i.e. for the entry ui in the output, after applying L it will become -(ui_xx + ui_yy)
    Detailed description of how it is constructed will be outlined in the documentation.ipynb file.

    Parameters
    ----------
    N : int
        The number of grid points in each dimension in the discretized frame.

    Returns
    -------
    L : numpy.ndarray
        numpy matrix of shape (N^2, N^2), that is the discretized version of negative laplacian operator.
    
    '''

    dx = 2/(N-1)
    C = Cn(N, dx, dx)
    CTC = C.T @ C
    L = np.eye(N**2, N**2)
    #construct interior points index 
    interior_index = []; boundary = [0, N-1]
    for i in range(N**2):
        if (i//N) in boundary or (i%N) in boundary:
            continue
        interior_index.append(i)
    B = L[interior_index, :]
    B[:, interior_index] = CTC
    L[interior_index, :] = B   
    return L

def H1Kernel(PINN, N, X_f_train, alpha=0):
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
        The number of grid data points in each dimension in the frame.
    X_f_train : numpy.ndarray
        numpy matrix of shape (N^2, 2) that contails all grid data points including BC points and colocation points
    alpha : float, default=0
        The damping factor that adds to the kernel

    Returns
    -------
    func : function 
        A function that takes a required input, of which the content is not important, and returns the H2 kernel matrix.
        of the specified PINN and damping factor. It is only supposed to be used in the newton-cg optimization framework.

    '''

    L = NegLaplacian(N)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(N**2)
        B =  tf.transpose(Jacobian)@(In + L)@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def H1semiKernel(PINN, N, X_f_train, alpha=0):
    L = NegLaplacian(N)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@L@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def HinvKernel(PINN, N, X_f_train, alpha=0):
    L = NegLaplacian(N)
    In = np.eye(N**2)
    LIninv = np.linalg.inv(In + L)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@LIninv@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def HinvsemiKernel(PINN, N, X_f_train, alpha):
    L = NegLaplacian(N)
    Linv = np.linalg.inv(L)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        B =  tf.transpose(Jacobian)@Linv@Jacobian
        return B + alpha*np.eye(numVars)
    return func

def FRKernel(PINN, X_f_train, alpha=0):
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        rho = np.array(prediction).flatten()
        numVars = Jacobian.shape[1]
        B = (tf.transpose(Jacobian)*(1/rho))@Jacobian
        return B + alpha*np.eye(numVars)
    return func

#W2 NGD kernel matrix 
def W2Kernel(PINN, N, alpha=0):
    dx = 2/(N-1)
    C = Cn(N, dx, dx)
    def func(X):
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(PINN.X_interior)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[PINN.X_interior.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        u = np.array(PINN.evaluate(PINN.X_interior)).flatten()
        u = np.hstack((u, u))
        B =  tf.transpose(Jacobian)@np.linalg.inv((C.T*u)@C)@Jacobian
        return B + alpha*np.eye(numVars)
    return func

##plotting methods
def solutionplot(u_pred, usol, savepath=None):
    #color map
    cmap = cm.get_cmap('jet')
    normalize = colors.Normalize(vmin=min_val, vmax=max_val)
    
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

#utility function 
def layertostr(layers):
    s = str(layers[0])
    for i in range(1, len(layers)):
        s += '-' + str(layers[i])
    return s
