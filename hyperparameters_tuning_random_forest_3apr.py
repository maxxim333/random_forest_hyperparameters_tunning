#All the modules
import collections, numpy
import pandas as pd
import numpy as np
import pandas_ml as pdml
import math
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import fbeta_score, make_scorer
import warnings
#For graphs
import matplotlib.pyplot as plt

#This defines a function
def hyperparams_tuning(dataset, message):

	#Load dataset
	features = pd.read_csv(dataset)
	labels = np.array(features['tag'])
	
	# Remove the labels from the features
	features= features.drop('tag', axis = 1)
	
	# Saving feature names for later use
	feature_list = list(features.columns)
	
	# Convert to numpy array
	features = np.array(features)
	
	# Using Skicit-learn to split data into training and testing sets
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
	
	#Create indices
	train_indices = [feature_list.index('substitytionmatrix'), feature_list.index('pssm-bative'), feature_list.index('entropy'), feature_list.index('SIFT'), feature_list.index('PPHDIV')]
	train_features = train_features[:, train_indices]
	test_features = test_features[:, train_indices]
	
	#The following block of code define ranges of hyperparameters
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 8, stop = 200, num = 192)]
	# Number of features to consider at every split
	max_features = ['auto', None]
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(5, 40, num = 35)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 3,4, 5, 6, 7, 8, 9]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1,2, 3,4, 5, 6, 7,8, 9, 10]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	#Class weight
	class_weight=[{0:1,1:1}]
	for z in range (2,10,1):
		class_weight.append({0:z,1:1})
	#Mean weight fraction leaf
	min_weight_fraction_leaf=[0.00,0.01,0.02,0.03,0.04]
	#Max leaf nodes
	max_leaf_nodes = [None]
	for z in range (5,200,5):
		max_leaf_nodes.append(z)
	#Min impurity decrease
	min_impurity_decrease = [0, 0.002, 0.004]
	
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap,
				   'class_weight': class_weight,
				   'min_weight_fraction_leaf': min_weight_fraction_leaf,
				   'max_leaf_nodes' : max_leaf_nodes,
				   'min_impurity_decrease':min_impurity_decrease
				   }
	

	
	#Next block of code will Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestClassifier()
	
	#Define scoring function (I want to maximize MCC)
	ftwo_scorer = make_scorer(matthews_corrcoef)
	
	# Random search of parameters, using 5 fold cross validation, 
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=0, random_state=42, n_jobs = -1, scoring=ftwo_scorer)

	# Fit the random search model
	rf_random.fit(train_features, train_labels)
	
	#Print values of parameters of the model with the best parameters
	print "Best Param for " + message
	print rf_random.best_params_
	print "----"
	
	
	#Compute performance metrics of the model
	#print "Best Random Performance for " + message
	best_random = rf_random.best_estimator_
	#print best_random
	predictions = best_random.predict(test_features)
	right=[]
	
	for i,j in zip(predictions,test_labels):
		if i==1 and j==1:
			right.append("TP")
		elif i==0 and j==0:
			right.append ("TN")
		elif i==1 and j==0:
			right.append("FP")
		elif i==0 and j==1:
			right.append("FN")
			
	a=collections.Counter(right)
	
	
	#Performance
	print message
	print "TP= ", (float(a["TP"]))
	print "TN= ", (float(a["TN"]))
	print "FP= ", (float(a["FP"]))
	print "FN= ", (float(a["FN"]))
	
	tp= (float(a["TP"]))
	tn= (float(a["TN"]))
	fp= (float(a["FP"]))
	fn= (float(a["FN"]))
	
	print "Accuracy= ",(float(a["TP"]) + float(a["TN"])) /(float(a["TP"]+a["TN"] + a["FP"] + a["FN"]))
	print "Sensitivity= ", (float(a["TP"])) / (float(a["TP"] + float(a["FN"]))) 
	print "Specificity= ", (float(a["TN"])) / (float(a["TN"] + float(a["FP"])))
	
	#MCC
	denominator=math.sqrt((tn+fp)*(tn+fn)*(tp+fp)*(tp+fn))
	if denominator!=0: mcc=(tp*tn-fp*fn)/denominator
	print "MCC= ", mcc
	print ""

	#Next bloc of code outputs importance of variables visualization
	importances = best_random.feature_importances_
	# List of tuples with variable and importance
	feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list[2:], importances)]

	# Sort the feature importances by most important first
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	# Print out the feature and importances 
	for pair in feature_importances:
		print('Variable: {:20} Importance: {}'.format(*pair))

	#Plot the importance
	plt.ion()

	# Set the style
	plt.style.use('fivethirtyeight')
	# list of x locations for plotting
	x_values = list(range(len(importances)))
	# Make a bar chart
	plt.bar(x_values, importances, orientation = 'vertical')
	# Tick labels for x axis
	plt.xticks(x_values, feature_list[2:], rotation='horizontal', fontsize=8)
	# Axis labels and title
	plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
	plt.tight_layout()
	plt.show()

	plt.savefig('/home/mvaskin/Desktop/github/random_forest/%s_importance.png' % (message))
	plt.close()


#Execute the functions for all the datasets
print "---------------------------------------------------------------------------------------------------"
print "Original Dataset"
hyperparams_tuning('/home/mvaskin/Desktop/github/random_forest/homologs.csv', "Homologs")
hyperparams_tuning('/home/mvaskin/Desktop/github/random_forest/non_congruent_siftpoly_known_orthologs.csv', "Orthologs")

