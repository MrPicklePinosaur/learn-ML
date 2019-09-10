import numpy as np
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN():

	K = 3

	def euc(self,a,b): #a and b are n dimensional vectors
		#Make sure a and b are the same length, if not, pad the shorter one
		tot = 0
		for i in range(len(b)):
			tot += (b[i]-a[i])**2
		return tot**0.5

	def nearest(self,row):
		#near = []
		min_dist = sys.maxsize
		min_index = 0
		for i in range(len(self.x_train)):
			dist = self.euc(row,self.x_train[i])
			if (dist < min_dist): #if we found a new lowest value
				min_dist = dist
				min_index = i
				#near.append(dist)
		#print(min_index)
		return self.y_train[min_index]

	def nearestK(self,row):
		dists = []
		for i in range(len(self.x_train)):
			dist = self.euc(row,self.x_train[i])
			dists.append((i,dist))
		dists.sort(key=lambda tup : tup[1]) #sort the array of dists from least to greatest dist

		#find the amount of occurences each label has
		occurs = {}
		for i in range(self.K): #Generates the labels of the k nearest neighbors
			label = self.y_train[dists[i][0]] #breaks when len(dists) < k
			if label in occurs: #If label is already in dict
				occurs[label] += 1 #increment occurences
			else:
				occurs[label] = 1 #create new occurence

		#Now find the most commonly occuring label
		max_key = max(occurs,key=occurs.get)
		return max_key


	def fit(self,x_train,y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self,x_test):
		predicts = [] 
		for row in x_test:
			predicts.append(self.nearestK(row))
		return predicts

'''
iris = load_iris()

#Split data into training and testing
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

#Create classifier
clf_KNN = KNN()

#Train clasifiers
clf_KNN.fit(x_train,y_train)

#Make predictions
KNN_result = clf_KNN.predict(x_test)
print(KNN_result)

#Determine accuracy
print("KNN accuracy:",accuracy_score(y_test,KNN_result))
'''