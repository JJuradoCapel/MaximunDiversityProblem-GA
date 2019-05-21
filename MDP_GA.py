import numpy as np
from examples.utils.txtParser import tableReader as tr
import sys, warnings, time

MAX_ITERATIONS_RANDOM_POPULATION = 50
THRESHOLD_FOR_WORSE_RESULTS = 300
MAX_EPOCH_WITHOUT_IMP = 30

class GeneticAlgorithm:

    def __init__(self, weightMatrix, m, initMethod = 'random', popSize = 500, parentSelectMethod = 'best', childPerParent = 2, initalMutationProb = 0.1, mutationDecay = 1, maxEpoch = 500, hybridParentsRatio = 0.5, verbose = 0):
        self.matrix = weightMatrix
        self.m = m
        self.n = weightMatrix.shape[0]
        self.popSize = popSize

        self.initMethod = initMethod

        self.parentSelectMethod = parentSelectMethod
        self.childPerParent = childPerParent
        self.mutationProb = initalMutationProb
        self.mutationDecay = mutationDecay

        self.maxEpoch = maxEpoch
        self.MAX_EPOCH_WITHOUT_IMP = MAX_EPOCH_WITHOUT_IMP
        self.hybridParentsRatio = hybridParentsRatio

        self.epoch = 0
        self.bestResult = 0
        self.bestResultEpoch = 0
        self.bestChoice = []

        self.verbose = verbose

        assert(weightMatrix.shape[0] == weightMatrix.shape[1]), "Matrix must be squared!"
        assert(self.n > m), "The sample must be smaller than the data!"
        assert(m%2 == 0), "The sample size must be even!"
        assert(self.popSize%(self.childPerParent + 2) == 0), "Wrong number of child per parents."
        
        if(not isinstance(weightMatrix,np.matrix)): weightMatrix = np.matrix(weightMatrix)

        self.resetPopulation()
        
        self.bestPop = self.pop
        self.distances = self.costFunction()
    
    def resetPopulation(self):
        self.pop = np.zeros((self.popSize,self.n))

        if self.initMethod == 'random':
            for i in range(self.popSize):
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
                
                a1 = np.where(np.logical_and(p1 == 1, ~eqBool))[0]
                a2 = np.where(np.logical_and(p2 == 1, ~eqBool))[0]

                if len(a1) < p1NewEle:
                    indexP1 = a1
                    indexP2 = np.random.choice(a2,size = notEqNum - len(a1), replace=False)
                elif len(a2) < p2NewEle:
                    indexP2 = a2
                    indexP1 = np.random.choice(a1,size = notEqNum - len(a2), replace=False)
                else:
                    indexP1 = np.random.choice(a1,size = int(p1NewEle), replace=False)
                    indexP2 = np.random.choice(a2,size = int(p2NewEle), replace=False)
                
                boolP1 = np.full(p1.size,False)
                boolP1[indexP1] = True

                boolP2 = np.full(p2.size,False)
                boolP2[indexP2] = True

                for j in range(p1.size):
                    if eqBool[j] or boolP1[j] or boolP2[j]:
                        childs[i,j] = 1
                    else:
                        childs[i,j] = 0

                if np.random.choice([True,False],p = [self.mutationProb, 1- self.mutationProb]):
                    #print("Mutation in epoch")
                    mut = np.random.choice(p1.size)
                    if childs[i,mut] == 0: 
                        childs[i,np.random.choice(np.where(childs[i] == 1)[0])] = 0
                        childs[i,mut] = 1
                    else:
                        childs[i,np.random.choice(np.where(childs[i] == 0)[0])] = 1
                        childs[i,mut] = 0
            return(childs)

        def bestSelect(N):
            return self.pop[sortedDistances[:N]]

        def wheelSelect(N):   
            parents = np.zeros((N,self.n))
            takedIndex = []
            total = sum(self.distances)
            for n in range(N):
                rnd = np.random.random()*total
                pointer = 0
                for i in sortedDistances:
                    pointer += self.distances[sortedDistances[i]]
                    if pointer > rnd and not i in takedIndex:
                        parents[n,:] = self.pop[i]
                        takedIndex.append(i)
                        break           
            return parents

        parentNumber = int((self.popSize/(self.childPerParent + 2)) * 2)
        sortedDistances = np.argsort(self.distances)[::-1]
        #print(self.distances[sortedDistances])

        if self.parentSelectMethod == 'best':
            parents = bestSelect(parentNumber)
        if self.parentSelectMethod == 'wheel':
            parents = wheelSelect(parentNumber)      
        if self.parentSelectMethod == 'hybrid':       
            bestPart = round(parentNumber*self.hybridParentsRatio)
            parents = bestSelect(bestPart)
            parents = np.append(parents, wheelSelect(parentNumber - bestPart),axis = 0)

        np.random.shuffle(parents)

        newPop = parents
        for i in range(0,parentNumber,2):
            childs = createChilds(parents[i],parents[i+1],self.childPerParent)
            newPop = np.append(newPop,childs, axis=0)

        self.mutationProb *= self.mutationDecay
        self.epoch += 1
        
        self.pop = newPop
        self.distances = self.costFunction()

        return None

    def run(self, timer = False):
        start = time.time()
        while self.epoch < self.maxEpoch:
            self.makeEpoch()
            sortedDistances = np.argsort(self.distances)[::-1]
            bestEpochResult = self.distances[sortedDistances[0]]
            diff = bestEpochResult - self.bestResult
            if self.verbose > 0:
                print("EPOCH: ",self.epoch)
                print("\tBest result in pop: ", bestEpochResult)
                print("\tImprovement: ",diff)
                print("\tEpochs without improvement: ",self.epoch - self.bestResultEpoch)
                print("-"*10,"\n")
            if diff > 0:
                self.bestResult = bestEpochResult
                self.bestResultEpoch = self.epoch
                self.bestChoice = self.pop[sortedDistances[0]]
                self.bestPop = self.pop
            if diff < -1*THRESHOLD_FOR_WORSE_RESULTS:
                self.pop = self.bestPop
                print("Warning! Population reset.") 
            if self.epoch - self.bestResultEpoch > self.MAX_EPOCH_WITHOUT_IMP:
                break
        end = time.time()
        return (self.bestChoice, self.bestResult, end - start) if timer else (self.bestChoice, self.bestResult)

    def reset(self):
        self.resetPopulation()

        self.epoch = 0
        self.bestResult = 0
        self.bestResultEpoch = 0
        self.bestChoice = []

        self.bestPop = self.pop
        self.distances = self.costFunction()

if __name__ == "__main__":
    n, m, matrix = tr('examples/data/GKD-c_1_n500_m50.txt')
    genetico = GeneticAlgorithm(matrix,m,parentSelectMethod='hybrid')
    print(genetico.run(timer = True))