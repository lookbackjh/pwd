# A code  for Pointwise Entropy Distributions for a Group Characteristic in High-dimensional and Sparse Microbiome Compositions

## How to use

---

run [main.py](http://main.py) to try various simulation, you can change various conditions based on your preference. 

You can test two types of simulation

1. Same-ratio(k) Sensitivity Analysis

set `parser.args.task` as 1 and by varying the level of `same_ratio` and number of features(`num_features`) you can check the p-value and its standard deviation after repeated permutation.

2. Sparse-level Sensitivity Analysis

set `parser.args.task` as 0  you are able to generate each group with various sparse ratio difference level, and you can check the p-value and its standard deviation. 

## Tutorial

---

from Example folder, you can see examples of synthetic data generation and  meta analysis of them, , origianl version of permutation with phylum-level microbiome, and plots of ordered pointwise entropy. Instructions are detailed in Jupyter notebook.