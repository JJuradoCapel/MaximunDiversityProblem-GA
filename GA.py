import numpy as np
import sys, warnings

MAX_ITERATIONS_RANDOM_POPULATION = 50
class Population:

    def __init__(self, weightMatrix, m, initMethod = 'random', popSize = 500, parentSelectMethod = 'best', childPerParent = 2, initalMutationProb = 0.05, mutationDecay = 0.8, maxEpoch = 500):
        self.matrix = weightMatrix
        self.m = m
        self.n = weightMatrix.shape[0]
        self.popSize = popSize
        self.childPerParent = childPerParent
        self.mutationProb = initalMutationProb
        self.mutationDecay = mutationDecay

        assert(weightMatrix.shape[0] == weightMatrix.shape[1]), "Matrix must be squared!"
        assert(self.n > m), "The sample must be smaller than the data!"
        assert(m%2 == 0), "The sample size must be even!"
        assert(self.popSize%(self.childPerParent + 2) == 0), "Wrong number of child per parents."
        
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
        
        self.distances = self.costFunction()
    
    def costFunction(self):
        distances = np.zeros(self.pop.shape[0])
        for i in range(distances.size):
            sample = self.pop[i][np.newaxis]
            dist = np.float_(np.dot(np.dot(sample,self.matrix),sample.T))/2
            distances[i] = dist
        return distances

    def makeEpoch(self):          

        def createChilds(p1,p2,n):

            childs = np.zeros((n,p1.size))
            eqBool = (p1 * p2) == 1
            notEqNum = self.m - sum(eqBool)
            prop = np.random.randint(1,10, size = n)/10

            for i in range(n):
                p1NewEle = np.round(notEqNum*prop[i])
                p2NewEle = notEqNum - p1NewEle
                
                indexP1 = np.random.choice(np.where(np.logical_and(p1 == 1, ~eqBool))[0],size = int(p1NewEle), replace=False)
                boolP1 = np.full(p1.size,False)
                boolP1[indexP1] = True

                indexP2 = np.random.choice(np.where(np.logical_and(p2 == 1, ~eqBool))[0],size = int(p2NewEle), replace=False)
                boolP2 = np.full(p2.size,False)
                boolP2[indexP2] = True

                for j in range(p1.size):
                    if eqBool[j] or boolP1[j] or boolP2[j]:
                        childs[i,j] = 1
                    else:
                        childs[i,j] = 0

                if np.random.choice([True,False],p = [self.mutationProb, 1- self.mutationProb]):
                    mut = np.random.choice(p1.size)
                    if childs[i,mut] == 0: 
                        childs[i,np.random.choice(np.where(childs[i] == 1)[0])] = 0
                        childs[i,mut] = 1
                    else:
                        childs[i,np.random.choice(np.where(childs[i] == 0)[0])] = 1
                        childs[i,mut] = 0
            return(childs)

        

        return createChilds(self.pop[1],self.pop[2], self.childPerParent)


if __name__ == "__main__":
    a = np.array([[0,1,2],[1,0,3],[2,3,0]])
    genetico = Population(a,2,popSize=3,childPerParent=1)
    print(genetico.makeEpoch())