import numpy as np

class GeneticAlgorithm():

    def __init__(self, weightMatrix, m, initMethod = 'random', popSize = 500 ):
        if weightMatrix.shape[0] != weightMatrix.shape[1]: 
            print("Matrix must be squared!")
            return
        if n > m:
            print("The sample must be smaller than the data!")
            return
        if m%2 != 0:
            print("The sample size must be even!")
            return
        
        self.matrix = weightMatrix
        self.m = m
        self.n = weightMatrix.shape[0]
        self.pop = np.ndarray((popSize,m))

        if initMethod == 'random':
            for i in range(popSize):
                ind = np.full(n,False)
                ind[np.random.choice(n,size=m,replace=False)] = True
                pop[i] = ind

        return self