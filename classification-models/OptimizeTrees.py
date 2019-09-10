#Find out what the optimal tree amount is for random forest

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import sys

from RForest import *

data = []

#Init models 
iris = load_iris()

#Split data into training and testing
x = iris.data
y = iris.target

NUM_OF_TESTS = 1 #how many tests per forest size
NUM_OF_FOREST_SIZES = 128 #how many forest sizes to try

for forest_size in range(1,NUM_OF_FOREST_SIZES+1):
	score = 0
	for i in range(NUM_OF_TESTS):

		#Randomly parse dataset
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

		clf_RForest = RForest(forest_size)
		clf_RForest.fit(x_train,y_train)

		score += accuracy_score(y_test,clf_RForest.predict(x_test))

	score /= NUM_OF_TESTS #average the test results
	data.append(score)

plt.plot([i for i in range(1,NUM_OF_FOREST_SIZES+1)],data)
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.show()