import numpy as np
import argparse
from src.Senstivity import Sensitivity
import copy
import math
from tqdm import tqdm

def datatransformation(w1):
    w2=copy.deepcopy(w1)
    n,d=np.shape(w2)
    if n==1:
         for j in range(d):
            if w2[0][j]==0:
                w2[0][j]=0
            else:
                w2[0][j]=math.log2(w2[0][j])
    else:
        for i in range(n):
            for j in range(d):
                if w2[i][j]==0:
                    w2[i][j]=0
                else:
                    w2[i][j]=math.log2(w2[i][j])
    return w2

def get_args():
    parser = argparse.ArgumentParser(description='Args Sparse Simulation')
    parser.add_argument('--num_feature',type=int,default=100,help='number of feature to simulate') ## only can choose n=100, 300, 600
    parser.add_argument('--p_num', type=int, default=10, help='seed')
    parser.add_argument('--repeat_num',type=int, default=10, help='to see the mean and sd for the p-value after repeated permutation')
    parser.add_argument('--x_grid_start',type=int, default=1,help='start of the x-axis')
    parser.add_argument('--x_grid_end',type=int, default=20,help='end of the x-axis')
    parser.add_argument('--same_ratio',type=int, default=100,help='Same ratio for each group') ## only can choose 10,50, 100 
    parser.add_argument('--interval',type=float,default=0.1,help='interval for x-axis' )
    parser.add_argument('--epsilon',type=float,default=0.0001,help='to_avoid zero division')
    parser.add_argument('--simulated',type=bool,default=True,help='False if you have your own prior knowledge for the absence raito') # if you have your own prior knowledge for the absence ratio set it to False

    parser.add_argument('--predefined',type=bool,default=True,help='False if there is predefined meta analysis') # if you have your own meta analysis for the data result set it to False
    parser.add_argument('--different_sparse_ratio',type=int, default=100, help="Default differnt ratio for sparse simulation problem set it to hundred. ")
    parser.add_argument('--task',type=int,default=0,help='0 for checking sensitivity for the sparsity ratio and 1 for checking sensitivity for various n, k') 
    args, _ = parser.parse_known_args()
    return args

def app_run(args):

    if args.simulated==True:
        feat_info=np.zeros(args.num_feature)
        feat_info+=1.5 ## by setting feat_info greater than one you do not consider the posterior or prior case.
    else:
        feat_info=np.loadtxt("your predefined prior information for each feature.txt") ## need to have your own prior information
    simulator=Sensitivity(args)
    simulator.simulate(feat_info)

def main():
 ## a dictionary to get the results for the sd and mean for the p-value (too obtain the standard deviation and mean for the p-value repeating 50 times of 1000 permutation)
    args=get_args()
    app_run(args) ##for simulation


if __name__ == "__main__":
    main()
