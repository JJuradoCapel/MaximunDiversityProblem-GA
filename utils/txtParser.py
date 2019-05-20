import numpy as np

def tableReader(filename):
    with open(filename) as f:
        a = f.readline()
        splitted = a.split(" ")
        n = int(splitted[0])
        m = int(splitted[1])

        weightMatrix = np.zeros((n,n))

        for line in f:
            splitted = line.split(" ")
            weightMatrix[int(splitted[0]),int(splitted[1])] = float(splitted[2])

        weightMatrix += weightMatrix.T
    
    return n, m, weightMatrix


if __name__ == "__main__":
    n,m,matrix = tableReader('GKD-c_1_n500_m50.txt')
    print(matrix.shape)