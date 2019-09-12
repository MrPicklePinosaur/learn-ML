import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import sys

from RForest import *
from DTree import *
from KNN import *

data = {"KNN" : [], "DTree" : [], "RForest" : []}

#Init models 
iris = load_iris()

#Split data into training and testing
x = iris.data
y = iris.target

AMOUNT_OF_TESTS = 8

for i in range(AMOUNT_OF_TESTS):

	#Create and train trees
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5) #split dataset

	clf_KNN = KNN()
	clf_KNN.fit(x_train,y_train)
	KNN_accuracy = accuracy_score(y_test,clf_KNN.predict(x_test))
	data["KNN"].append(KNN_accuracy)

	clf_DTree = DTree()
	clf_DTree.fit(x_train,y_train)
	DTree_accuracy = accuracy_score(y_test,clf_DTree.predict(x_test))
	data["DTree"].append(DTree_accuracy)

	clf_RForest = RForest(64)
	clf_RForest.fit(x_train,y_train)
	RForest_accuracy = accuracy_score(y_test,clf_RForest.predict(x_test))
	data["RForest"].append(RForest_accuracy)

for algo in data:
	acc = np.sum(data[algo])/len(data[algo])
	print(algo,acc)



