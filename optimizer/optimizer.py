import time
import numpy as np
import tensorflow as tf

class MiniBatch:
    '''
    MiniBatch utility object class. 

    ...
    Attributes
    ----------
    X_f_train : numpy.ndarray
        numpy 2d array of size (N^2, 2), where N is the number of data points in each dimension.
        It is the entire discretized data points. 
    X_u_train : numpy.ndarray
        numpy 2d array of size (4N-4, 2), where N is the number of data points in each dimension.
        It is the boundary data points. 
    u_train : numpy.ndarray
        numpy 1d array of size (4N-4,), where N is the number of  data points in each dimension.
        It is the boundary condition, the truth value of the function on the boundary data points. 
    BC_ratio : float
        The percentage of the boundary points to include in the mini batch.
    Interior_ratio : float
        The percentage of the interior data points to include in the mini batch.
    BC_size : int
        The number of boundary points to include in the mini batch
    Interior_size : int
        The number of interior points to include in the mini batch
    BC_index : numpy.ndarray
        Numpy 1d array of the indices of boundary points in the X_f_train array.
    Interior_index : numpy.ndarray
        Numpy 1d array of the indices of interior points in the X_f_train array.
    N : int
        The number of data points in each dimension.

    Methods
    -------
    sample()
        Constructs and returns a randomly generated mini batch of dataset given the BC_ratio and interior_ratio.
    
    '''

    def __init__(self, X_f_train, X_u_train, u_train, BC_ratio, Interior_ratio):
        '''
        Constructor for MiniBatch

        X_f_train : numpy.ndarray
            the entire train points
        X_u_train : numpy.ndarray
            boundary points
        u_train : numpy.ndarray
            1d numpy array of values on the boundary points
        BC_ratio: float
            the ratio of boundary points to include in the mini batch
        Interior_ratio: float
            the ratio of the interior points to include in the mini batch

        '''

        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train = u_train
        self.BC_ratio = BC_ratio
        self.Interior_ratio = Interior_ratio
        self.BC_size = round(len(X_u_train)*BC_ratio)
        self.Interior_size = round((len(X_f_train)-len(X_u_train))*Interior_ratio)
        self.BC_index = [] ##indices of Boundary points
        self.N = int(len(X_f_train)**.5)
        self.Interior_index = [] ##indices of colocation points
        for i in range(self.N**2):
            if (i//self.N) in [0, self.N-1] or (i%self.N) in [0, self.N-1]:
                self.BC_index.append(i)
                continue
            self.Interior_index.append(i)
        self.BC_index = np.array(self.BC_index)
        self.Interior_index = np.array(self.Interior_index)

    ##return a minibatch sample of specified ratio
    def sample(self):
        '''
        Sample method that constructs and returns a mini batch dataset. 
        According to the given BC ratio and interior ratio.

        Returns
        -------
        tuple
            A tuple of length 6.
            first element: 2d numpy array of the entire selected mini batch data points.
            second element: 2d numpy array of the selected boundary points in the mini batch.
            third element: 1d numpy array of the truth values of the selected boundary points in the mini batch.
            forth element: 1d numpy array of the indices of the selected data points in the X_f_train array.
            fifth element: 1d numpy array of the indices of the selected boundary points in the X_f_train array.
            sixth element: 1d numpy array of the indices of the selected interior points in the X_f_train array.
        
        '''

        ##figure out the index
        BC_sample_index = np.random.choice(self.BC_index, self.BC_size, replace=False)
        BC_sample_index.sort()
        Interior_sample_index = np.random.choice(self.Interior_index, self.Interior_size, replace=False)
        Interior_sample_index.sort()
        sample_index = np.concatenate([BC_sample_index, Interior_sample_index])
        sample_index.sort()
        ##construct the entire minibatch
        X_f_train = self.X_f_train[sample_index]
        ##construct the BC minibatch
        X_u_train = self.X_f_train[BC_sample_index]
        ##construct the bounday value of the mini batch on boundary
        u_train = self.u_train[:self.BC_size]
        return (X_f_train, X_u_train, u_train, sample_index, BC_sample_index, Interior_sample_index)

##this method does Stochastic Gradient Descent on the give PINN model
##it uses backtracking line search to do the GD
def SGD_optimizer(PINN, lr, n_iter, BC_ratio=1, Interior_ratio=1):
    '''
    This methods performs stochastic gradient descent on a given PINN.
    Instead of fixing the step size, this method uses backtracking line search to modify the step size.
    Given a step size upperbound, the step size halves until the new loss is reduced. 
    The method records the loss trace and cpu time in the PINN.loss_trace and PINN.clock attributes. 

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model on which to perform the SGD.
    lr : float
        The upperbound for the step size.
    n_iter : int
        The number of iteratons of SGD to perform.
    BC_ratio : float, default=1
        The ratio of boundary points to include in each mini batch.
    Interior_ratio : float, default=1
        The ratio of interior data points to include in each mini batch 
    
    '''

    ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, _, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = grads_1d.numpy()

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter - step*grads_1d
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter - step*grads_1d
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)
        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace)-1, loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##it uses backtracking line search 
def L2_optimizer(PINN, lr, n_iter, BC_ratio=1, Interior_ratio=1, alpha=0):
    '''
    This methods performs stochastic L2 natural gradient descent on a given PINN.
    Instead of fixing the step size, this method uses backtracking line search to modify the step size.
    Given a step size upperbound, the step size halves until the new loss is reduced. 
    The method records the loss trace and cpu time in the PINN.loss_trace and PINN.clock attributes. 

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model on which to perform the algorithm.
    lr : float
        The upperbound for the step size.
    n_iter : int
        The number of iteratons to perform.
    BC_ratio : float, default=1
        The ratio of boundary points to include in each mini batch.
    Interior_ratio : float, default=1
        The ratio of interior data points to include in each mini batch 
    alpha : float, default=0
        The damping factor to add in the L2 kernel.
        It is recommended to set a non-zero value when the number of data points in the mini batch 
        is less then the number of parameters. 
    
    '''

    ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, _, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the L2 kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@Jacobian
        ##compute the L2 descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##H1 NGD 
##it uses backtracking line search 
def H1_optimizer(PINN, lr, n_iter, L, BC_ratio=1, Interior_ratio=1, alpha=0):
    '''
    This methods performs stochastic H1 natural gradient descent on a given PINN.
    Instead of fixing the step size, this method uses backtracking line search to modify the step size.
    Given a step size upperbound, the step size halves until the new loss is reduced. 
    The method records the loss trace and cpu time in the PINN.loss_trace and PINN.clock attributes. 

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model on which to perform the algorithm.
    lr : float
        The upperbound for the step size.
    n_iter : int
        The number of iteratons to perform.
    L : numpy.ndarray
        2d numpy array of the discretized negative laplacian operator matrix
    BC_ratio : float, default=1
        The ratio of boundary points to include in each mini batch.
    Interior_ratio : float, default=1
        The ratio of interior data points to include in each mini batch 
    alpha : float, default=0
        The damping factor to add in the H1 kernel.
        It is recommended to set a non-zero value when the number of data points in the mini batch 
        is less then the number of parameters. 
    
    '''

    ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##compute the I + L
    N = int(PINN.X_f_train.shape[0]**0.5)
    L = L + np.eye(N**2)
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the H1 kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@L[sample_index][:,sample_index]@Jacobian
        ##compute the H1 descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##H1 seminorm NGD 
##it uses backtracking line search 
def H1semi_optimizer(PINN, lr, n_iter, L, BC_ratio=1, Interior_ratio=1, alpha=0):
    '''
    This methods performs stochastic H1 seminorm natural gradient descent on a given PINN.
    Instead of fixing the step size, this method uses backtracking line search to modify the step size.
    Given a step size upperbound, the step size halves until the new loss is reduced. 
    The method records the loss trace and cpu time in the PINN.loss_trace and PINN.clock attributes. 

    Parameters
    ----------
    PINN : Sequentialmodel
        The PINN model on which to perform the algorithm.
    lr : float
        The upperbound for the step size.
    n_iter : int
        The number of iteratons to perform.
    L : numpy.ndarray
        2d numpy array of the discretized negative laplacian operator matrix
    BC_ratio : float, default=1
        The ratio of boundary points to include in each mini batch.
    Interior_ratio : float, default=1
        The ratio of interior data points to include in each mini batch 
    alpha : float, default=0
        The damping factor to add in the H1 seminorm kernel.
        It is recommended to set a non-zero value when the number of data points in the mini batch 
        is less then the number of parameters. 
    
    '''

    ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the H1 seminorm kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@L[sample_index][:,sample_index]@Jacobian
        ##compute the H1 seminorm descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)
    

##H-1 NGD 
##it uses backtracking line search 
def Hinv_optimizer(PINN, lr, n_iter, L, BC_ratio=1, Interior_ratio=1, alpha=0):
     ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##compute the (I + L)^-1
    N = int(PINN.X_f_train.shape[0]**0.5)
    L = np.linalg.inv(L + np.eye(N**2))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the H-1 kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@L[sample_index][:,sample_index]@Jacobian
        ##compute the H-1 descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##H-1 seminorm NGD 
##it uses backtracking line search 
def Hinvsemi_optimizer(PINN, lr, n_iter, L, BC_ratio=1, Interior_ratio=1, alpha=0):
     ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##compute the L^-1
    L = np.linalg.inv(L)
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the H-1 seminorm kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@L[sample_index][:,sample_index]@Jacobian
        ##compute the H-1 seminorm descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##Fisher-Rao Natural Gradient Descent
##it uses backtracking line search 
def FR_optimizer(PINN, lr, n_iter, BC_ratio=1, Interior_ratio=1, alpha=0):
     ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the FR NGD kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        rho = np.array(prediction).flatten()
        In = np.eye(numVars)
        kernel =  alpha*In + (tf.transpose(Jacobian)*(1/rho))@Jacobian
        ##compute the FR NGD descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)

##W2 Natural Gradient Descent
##it uses backtracking line search 
def W2_optimizer(PINN, lr, n_iter, C, BC_ratio=1, Interior_ratio=1, alpha=0):
     ##construct the minibatch object
    minibatch = MiniBatch(PINN.X_f_train, PINN.X_u_train, PINN.u_train, BC_ratio, Interior_ratio)
    ##append the initial loss to the loss_trace list
    init_loss, init_u, init_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=True)
    tf.print('Initial total loss: {}, COLO: {}, BC: {}'.format(init_loss, init_u, init_f))
    ##enter the loop
    for i in range(n_iter):
        ##construct a minibatch set
        X_f_train, X_u_train, u_train, sample_index, _, _ = minibatch.sample()
        #compute the gradient using this minibatch
        with tf.GradientTape() as tape:
            tape.watch(PINN.trainable_variables)
            loss_val, loss_u, loss_f = PINN.loss(X_u_train, u_train, X_f_train) 
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        del tape 
        grads_1d = [ ] #store 1d grads 
        for j in range (len(PINN.layers)-1):
            grads_w_1d = tf.reshape(grads[2*j],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*j+1],[-1]) #flatten biases
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
        grads_1d = -grads_1d.numpy()
        ##compute the W2 NGD kernel matrix
        with tf.GradientTape(persistent=True) as tape:
            prediction = PINN.evaluate(X_f_train)
        Jac = tape.jacobian(prediction, PINN.trainable_variables, experimental_use_pfor=False)
        Jacobian = tf.concat([tf.reshape(Jac[i],[X_f_train.shape[0], -1]) for i in range(len(Jac))],axis=1)
        numVars = Jacobian.shape[1]
        u = np.array(PINN.evaluate(PINN.X_interior)).flatten()
        u = np.hstack((u, u))
        B = np.linalg.inv((C.T*u)@C)
        In = np.eye(numVars)
        kernel =  alpha*In + tf.transpose(Jacobian)@B[sample_index][:,sample_index]@Jacobian
        ##compute the W2 NGD descent direction 
        direction = np.linalg.inv(kernel)@grads_1d

        oldParameter = PINN.get_weights().numpy()
        step = lr
        ##compute the new parameter, doing the descent operation
        newParameter = oldParameter + step*direction
        ##set the new parameter
        PINN.set_weights(newParameter)
        ##compute and record the total loss
        loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        ##backtracking line search
        i = 0
        ##takes about 0.05 sec each backtracking iteration
        while (loss_value > PINN.loss_trace[-1]):
            i+=1
            step /= 2
            newParameter = oldParameter + step*direction
            PINN.set_weights(newParameter)
            loss_value, loss_u, loss_f = PINN.loss(PINN.X_u_train, PINN.u_train, PINN.X_f_train, record=False)
        PINN.loss_trace.append(loss_value)

        tf.print('{}th iteration: total loss: {}, COLO: {}, BC: {}, backtracking #: {}'.format(len(PINN.loss_trace), loss_value, loss_f, loss_u, i))
        ##record cpu time
        PINN.clock.append(time.time()-PINN.start_time)