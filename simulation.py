import argparse
parser = argparse.ArgumentParser(description='Args Sparse Simulation')
parser.add_argument('--p_num', type=int, default=10, help='num ber of permutation')
parser.add_argument('--x_grid_start',type=int, default=1,help='start of the x-axis')
parser.add_argument('--x_grid_end',type=int, default=20,help='end of the x-axis')
parser.add_argument('--same_ratio',type=int, default=90,help='Difference ratio for each group') ## only can choose 50,90, 100
parser.add_argument('--interval',type=float,default=0.1,help='interval for x-axis' )
parser.add_argument('--epsilon',type=float,default=0.0001,help='to_avoid zero division')
args, _ = parser.parse_known_args()

import pandas as pd
import numpy as np
otu=pd.read_csv("otu.csv")

otu['c']
# drop largest 
otu['c'].drop(otu['c'].idxmax())
# divide by sum of the row
ratio=otu['c']/otu['c'].sum()

# please decrease the ratio of the first half by 0.5 and increase the ratio for decreased amount to the second half
decreased=ratio[:int(len(ratio)/2)]-ratio[:int(len(ratio)/2)]*0.5

increased=decreased.sum()/len(ratio[int(len(ratio)/2):])
ratio_first_half=ratio[:int(len(ratio)/2)]-decreased
ratio_second_half=ratio[int(len(ratio)/2):]+increased
ratio_temp=pd.concat([ratio_first_half,ratio_second_half])
g1=np.random.dirichlet(otu['c'],size=100)
g2=np.random.dirichlet(ratio_temp*1500,size=100)

vacteriainfo=np.arange(233)
pointwiseentropy_1=-np.log(g1.mean(axis=0))
pointwiseentropy_2=-np.log(g2.mean(axis=0))
#change row and column
p1=-np.log(g1)
p2=-np.log(g2)
p1=p1.T
p2=p2.T
feat_info=np.ones((233,1))
feat_info=feat_info+0.5

import math
import numpy as np
import matplotlib.pylab as plt
from src.Util import Permutator
import time
from tqdm import tqdm
from tqdm import tqdm
pvals=[]
for i in tqdm(range(10)):
    p=Permutator(p1,p2, feat_info,args )
    pvals.append(p.permutation())