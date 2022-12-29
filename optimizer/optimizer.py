
import numpy as np

class MiniBatch:
    def __init__(self, X_f_train, X_u_train, u_train, BC_ratio, Interior_ratio):
        '''
        Constructor for MiniBatch
        X_f_train: the entire train points
        X_u_train: boundary points
        u_train: value on the boundary points
        BC_ratio: the ratio of boundary points to include in the mini batch
        Interior_ratio: the ratio of the interior points to include in the mini batch
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



if __name__ == '__main__':
    print('hello')