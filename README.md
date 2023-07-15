# A code  for Pointwise Entropy Distributions for a Group Characteristic in High-dimensional and Sparse Microbiome Compositions

## How to use

---

run [main.py](http://main.py) to try various simulation, you can change various conditions based on your preference. 

You can test two types of simulation

1. Same-ratio(k) Sensitivity Analysis
set `parser.args.task` as 1 and by varying the level of `same_ratio` and number of features(`num_features`) you can check the p-value and its standard deviation after repeated permutaion.
2. Sparse-level Sensitivity Analysis
set `parser.args.task` as 0 and by varying the level of  `different_sparse_ratio` in parser.args, you can check the p-value and its standard deviation for various sparse ratio difference generated for each group. 

## Tutorial

---

from Example folder, you can see examples for meta analysis, origianl version of permutation with phylum-level microbiome, and plotting ordered pointwise entropy. Instructions are detailed in Jupyter notebook.