# random_forest_hyperparameters_tunning
In this work I use Random Forest tool on two datasets and find the optimal hyperparameters for the algorithm on both datasets and compare their performance.

# Usage
I first begin to define a grid with a range of hyperparameters. Then, 100 iterations of train/crossvalidation/performance metrics calculation are performed. The algorithm with the combination yielding the maximal MCC (Matthews Coefficient) is then chosen to be the best. The list of hyperparameters of the best estimator is printed in the screen, along with the list of parameters used and 8 performance metrics (including maximized MCC). 

Also, for each of the dataset, the importance of variables is printed on screen and a graph is generated (an example of graph is included).

All is done automatically, only by changing the path to the two .csv files in the last two lines of the script.

#Dataset and Goal
As in previous case, this is the dataset of protein polymorphisms with different attributes regarding its conservation patterns in homologs or orthologs (depending on the dataset) and biophysical properties. The goal is to compare which dataset will yield the best predictor. 

The dataset is heavily manipulated with random noise inserted in each of the values.

# Example of Output and Interpretation:

```
Original Dataset
Best Param for Homologs
{'min_impurity_decrease': 0, 'max_leaf_nodes': 30, 'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 28, 'max_features': 'auto', 'min_weight_fraction_leaf': 0.02, 'min_samples_split': 8, 'max_depth': 36, 'class_weight': {0: 4, 1: 1}}
----
Homologs
TP=  397.0
TN=  79.0
FP=  73.0
FN=  125.0
Accuracy=  0.706231454006
Sensitivity=  0.760536398467
Specificity=  0.519736842105
MCC=  0.25496153652

Variable: pssm-bative          Importance: 0.27182
Variable: PPHDIV               Importance: 0.26774
Variable: entropy              Importance: 0.21795
Variable: SIFT                 Importance: 0.14139
Variable: substitytionmatrix   Importance: 0.1011
Best Param for Orthologs
{'min_impurity_decrease': 0.002, 'max_leaf_nodes': 95, 'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 150, 'max_features': None, 'min_weight_fraction_leaf': 0.01, 'min_samples_split': 3, 'max_depth': 11, 'class_weight': {0: 2, 1: 1}}
----
Orthologs
TP=  534.0
TN=  67.0
FP=  67.0
FN=  6.0
Accuracy=  0.891691394659
Sensitivity=  0.988888888889
Specificity=  0.5
MCC=  0.627857057104

Variable: SIFT                 Importance: 0.51102
Variable: PPHDIV               Importance: 0.3075
Variable: pssm-bative          Importance: 0.08396
Variable: substitytionmatrix   Importance: 0.05619
Variable: entropy              Importance: 0.04133
```

Original Dataset (Homologs) yields the best predictor with MCC of 0.25 for which the parameters are specified inside the brackets. The most important variable is pssm-bative

The orthologs dataset yields the best predictor with a much higher MCC than its homologs counterpart and in this case the most important variable is SIFT.
