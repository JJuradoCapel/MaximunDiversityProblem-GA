from utils.txtParser import tableReader
import numpy as np
import os, sys

sys.path.append("..")
from MDP_GA import GeneticAlgorithm as GA

dataFiles = os.listdir("data")

for table in dataFiles:
    results = []
    n,m,matrix = tableReader("data/" + table)
    for seed in range(1000,1021):
        np.random.seed(seed)
        alg = GA(matrix,m,verbose=1)
        _,cost,time = alg.run(timer=True)
        results.append([cost, time])

    with open("results/" + table, 'w') as f:
        f.writelines(["%s\n" % item  for item in results])
        f.close()
