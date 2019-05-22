from utils.txtParser import tableReader
import numpy as np
import os, sys

sys.path.append("..")
from MDP_GA import GeneticAlgorithm as GA

def runAlgorithm(matrix,m,params):
    np.random.seed(1000)
    alg = GA(matrix,m,verbose=1,**params)
    _,cost,time = alg.run(timer=True)

    return cost,time

n,m,matrix = tableReader('data/GKD-c_1_n500_m50.txt')

# Metodo de selección de padres
methods = ['best','wheel','hybrid']
results = []

for method in methods:
    if method == 'hybrid':
        for ratio in [0.25,0.75,0.5]:
            cost,time = runAlgorithm(matrix,m,{'parentSelectMethod':method,'hybridParentsRatio':ratio})
            results.append([method + '-ratio=' + str(ratio),cost,time])
    else:
        cost,time = runAlgorithm(matrix,m,{'parentSelectMethod':method})
        results.append([method,cost,time])

with open("results/methods.txt", 'w') as f:
    f.writelines(["%s\n" % item  for item in results])
    f.close()

# Size of the population
pops = [200,300,500,700,1000,1500]
results = []

for pop in pops:
    cost,time = runAlgorithm(matrix,m,{'popSize':pop})
    results.append(['population=' + str(pops),cost,time])

with open("results/pops.txt", 'w') as f:
    f.writelines(["%s\n" % item  for item in results])
    f.close()

# Mutations
prop = [0.01, 0.05, 0.1, 0.2]
ratios = [1,0.9,0.8,0,6]
results = []

for prop in props:
    for ratio in ratios:
        cost,time = runAlgorithm(matrix,m,{'initalMutationProb':prop, 'mutationDecay': ratio})
        results.append(['prop=' + str(prop) + '-ratio=' + str(ratio),cost,time])

with open("results/mutation.txt", 'w') as f:
    f.writelines(["%s\n" % item  for item in results])
    f.close()

# Children per parent
childs = [2,3,8]
results = []

for child in childs:
    cost,time = runAlgorithm(matrix,m,{'childPerParent':child})
    results.append([child,cost,time])

with open("results/childs.txt", 'w') as f:
    f.writelines(["%s\n" % item  for item in results])
    f.close()
