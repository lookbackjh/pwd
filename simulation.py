import argparse
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import numpy as np
import argparse
from src.Senstivity import Sensitivity
import copy
import math
from tqdm import tqdm
from src.Util import Permutator
import time
from tqdm import tqdm
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Args Sparse Simulation')
parser.add_argument('--p_num', type=int, default=1000, help='num ber of permutation')
parser.add_argument('--x_grid_start',type=int, default=1,help='start of the x-axis')
parser.add_argument('--x_grid_end',type=int, default=20,help='end of the x-axis')
parser.add_argument('--same_ratio',type=int, default=90,help='Difference ratio for each group') ## only can choose 50,90, 100
parser.add_argument('--interval',type=float,default=0.1,help='interval for x-axis' )
parser.add_argument('--epsilon',type=float,default=0.0001,help='to_avoid zero division')
parser.add_argument('--repeat_num',type=int, default=1, help='to see the mean and sd for the p-value after repeated permutation')
parser.add_argument('--num_feature',type=int,default=100,help='number of feature to simulate') ## only can choose n=100, 300, 600
parser.add_argument('--predefined',type=bool,default=False,help='False if there is predefined meta analysis')
args, _ = parser.parse_known_args()


beta=0.7
# randomly select the number of microiomes to change the ratio
num_microbiomes_to_change=200
# sort
sort_option=False



otu=pd.read_csv("Example/Meta_Analysis_Example/throatotu_csv.csv")

otu['x']
# drop three most abundant species

#sort 
otu=otu.sort_values(by='x',ascending=False)
otu=otu.iloc[3:]



# divide by sum of the row
ratio=otu['x']/otu['x'].sum()


# randomly select the microbiomes to change the ratio
ratio=ratio.to_numpy()
if sort_option ==False:
    # sort ratio
    np.random.shuffle(ratio)




ratio_to_change=ratio[:num_microbiomes_to_change]
ratio_to_change
ratio_fixed=ratio[num_microbiomes_to_change:]

decreased=ratio_to_change[:int(len(ratio_to_change)/2)]-ratio_to_change[:int(len(ratio_to_change)/2)]*beta

ratio_change_first_half=ratio[:int(len(ratio_to_change)/2)]-decreased
ratio_second_half=ratio_to_change[int(len(ratio_to_change)/2):]
increased=np.zeros(len(ratio_second_half))
sum_ratio_second_half=ratio_second_half.sum()   

for i in range(len(ratio_second_half)):

    increased[i]=(decreased.sum()*ratio_second_half[i])/sum_ratio_second_half
# distribute the increased ratio to the second half with equivalent to 

ratio_second_half=ratio_to_change[int(len(ratio_to_change)/2):]+increased

g1_param=ratio
g_changed=np.concatenate([ratio_change_first_half,ratio_second_half])
g2_param=np.concatenate([g_changed,ratio_fixed])

theta=0.0001 # overdispersion parameter
num_samples=50 # number of samples.  this will be a major paramter. 

g1=np.random.dirichlet(g1_param*(1/theta),size=1)
g2=np.random.dirichlet(g2_param*(1/theta), size=1)
# s
#create count poisson
g1_count_total=np.random.poisson(10000.0,num_samples)
g2_count_total=np.random.poisson(10000.0,num_samples)

#what i watn to do is get dividing rowwise summation
# g1_normalized = g1_count / g1_count.sum(axis=1)[:, np.newaxis]
# g2_normalized = g2_count / g2_count.sum(axis=1)[:, np.newaxis]
g1_count=np.zeros((num_samples,len(g1_param)))
g2_count=np.zeros((num_samples,len(g2_param)))

for i in range(num_samples):
    g1_count[i]=np.random.multinomial(g1_count_total[i],g1[0])
    g2_count[i]=np.random.multinomial(g2_count_total[i],g2[0])

g1_normalized = g1_count / g1_count.sum(axis=1)[:, np.newaxis]
g2_normalized = g2_count / g2_count.sum(axis=1)[:, np.newaxis]

epsilon = 1e-10  # Small constant to avoid log(0)
g1_entropy = np.where(g1_normalized > 0.0001, -np.log(g1_normalized ), 0)
g2_entropy = np.where(g2_normalized > 0.0001, -np.log(g2_normalized ), 0)

g1_entropy = g1_entropy.T
g2_entropy = g2_entropy.T
pvals=[]

x_grid=np.arange(0,15,0.1)
feat_info=np.zeros(g1_entropy.shape[0])
feat_info+=1.1 ## as this is simulated  case, you cannot consider the prior-posterior case. 
args.num_feature=g1_entropy.shape[0]

simulator=Sensitivity(args)

    #p1_100,p2_100=simulator.get_group_sparse(args.num_feature,k)
p=Permutator(g1_entropy,g2_entropy,feat_info,args)
pval=p.metapermutation()

