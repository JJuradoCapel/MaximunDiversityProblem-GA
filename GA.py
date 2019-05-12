import numpy as np
import sys, warnings

MAX_ITERATIONS_RANDOM_POPULATION = 50
class Population:

    def __init__(self, weightMatrix, m, initMethod = 'random', popSize = 500 ):
        self.matrix = weightMatrix
        self.m = m
        self.n = weightMatrix.shape[0]

        assert(weightMatrix.shape[0] == weightMatrix.shape[1]), "Matrix must be squared!"
        assert(self.n > m), "The sample must be smaller than the data!"
        assert(m%2 == 0), "The sample size must be even!"
        
        if(not isinstance(weightMatrix,np.matrix)): weightMatrix = np.matrix(weightMatrix)

        self.pop = np.zeros((popSize,self.n))


        if initMethod == 'random':
            for i in range(popSize):
                j = 0
                while True:
                    ind = np.full(self.n,0)
                    ind[np.random.choice(self.n,size=m,replace=False)] = 1

                    if sum((self.pop == tuple(ind)).all(axis = 1)) == 0 : break

                    if i > MAX_ITERATIONS_RANDOM_POPULATION:
                        warnings.warn("Got maximum number of iteration in the random initial population. There is a repeted sample in the population. Please, select another method or reduce the population size.")
                        break
                    j += 1

                self.pop[i,:] = ind
        
        self.distances = costFunction()
    
    def costFunction(self):
        distances = np.zeros(self.pop.shape[0])
        for i in range(distances.size):
            sample = self.pop[i][np.newaxis]
            dist = np.float_(np.dot(np.dot(sample,self.matrix),sample.T))/2
            distances[i] = dist
        return distances

if __name__ == "__main__":

    a = np.array([[0,1,2],[1,0,3],[2,3,0]])
    genetico = Population(a,2,popSize=3)
    print(genetico.pop,genetico.costFunction())