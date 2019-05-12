import numpy as np
import sys

class GeneticAlgorithm:

    def __init__(self, weightMatrix, m, initMethod = 'random', popSize = 500 ):
        self.matrix = weightMatrix
        self.m = m
        self.n = weightMatrix.shape[0]

        assert(weightMatrix.shape[0] == weightMatrix.shape[1]), "Matrix must be squared!"
        assert(self.n > m), "The sample must be smaller than the data!"
        assert(m%2 == 0), "The sample size must be even!"
        
        if(not isinstance(weightMatrix,np.matrix)): weightMatrix = np.matrix(weightMatrix)

        self.pop = np.ndarray((popSize,m))

        if initMethod == 'random':
            for i in range(popSize):
                ind = np.full(self.n,0)
                ind[np.random.choice(self.n,size=m,replace=False)] = 1
                self.pop[i] = ind
    
    def costFunction(self, sample, matrix):
        if len(sample.shape) == 1: sample = sample[np.newaxis]
        dist = np.float_(sample * matrix * sample.T)/2