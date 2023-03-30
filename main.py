from scipy.stats import nbinom,truncnorm
import numpy as np
import argparse
from collections import defaultdict
from src.Senstivity import Sensitivity
def get_args():
    parser = argparse.ArgumentParser(description='Args Sparse Simulation')
    parser.add_argument('--num_feature',type=int,default=300,help='number of feature to simulate') ## only can choose n=100, 300, 600
    parser.add_argument('--p_num', type=int, default=100, help='seed')
    parser.add_argument('--repeat_num',type=int, default=50, help='to see the mean and sd for the p-value after repeated permutation')
    parser.add_argument('--x_grid_start',type=int, default=1,help='start of the x-axis')
    parser.add_argument('--x_grid_end',type=int, default=20,help='end of the x-axis')
    parser.add_argument('--same_ratio',type=int, default=90,help='Difference ratio for each group') ## only can choose 50,90, 100
    parser.add_argument('--interval',type=float,default=0.1,help='interval for x-axis' )
    parser.add_argument('--epsilon',type=float,default=0.0001,help='to_avoid zero division')
    parser.add_argument('--simulated',type=bool,default=True,help='False if you have your own prior knowledg for the absence raito') ## 나중에 건드려야 할듯.
    parser.add_argument('--predefined',type=bool,default=True,help='False if there is predefined meta analysis') ## 나중에 건드려야 할듯.
    parser.add_argument('--different_sparse_ratio',type=int, default=100, help="Default differnt ratio for sparse simulation problem set it to hundred. ")
    parser.add_argument('--task',type=int,default=1,help='0 for checking sensitivity for the sparsity ratio and 1 for checking sensitivity for various n, k') 
    args, _ = parser.parse_known_args()
    return args
def app_run(args):

    k_candidate=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    d = defaultdict(list) ## a dictionary to get the results for the sd and mean for the p-value (too obtain the standard deviation and mean for the p-value repeating 50 times of 1000 permutation)
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
    app_run(args)

if __name__ == "__main__":
    main()
