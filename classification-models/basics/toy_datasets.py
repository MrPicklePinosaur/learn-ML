import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()

'''
testing_index = [0,50,100] #remove some data for later testing
train_data = np.delete(iris.data,testing_index,axis=0)
train_target = np.delete(iris.target,testing_index)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#testing
print("Expected results",iris.target[testing_index])
print("Predicted",clf.predict(iris.data[testing_index]))
'''

#Split data into training and testing
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

#Create classifier
clf_tree = tree.DecisionTreeClassifier()
clf_KNN = KNeighborsClassifier()

#Train clasifiers
clf_tree.fit(x_train,y_train)
clf_KNN.fit(x_train,y_train)

#Make predictions
tree_result = clf_tree.predict(x_test)
KNN_result = clf_KNN.predict(x_test)

#Determine accuracy
print("tree accuracy:",accuracy_score(y_test,tree_result))
print("KNN accuracy:",accuracy_score(y_test,KNN_result))
