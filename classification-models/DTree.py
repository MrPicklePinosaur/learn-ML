import numpy as np
import sys
from pprint import *

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DTree:

	def __init__(self):
		self.root = None

	def fit(self,x_train,y_train): #this is where we build the tree
		
		#Find the best question to ask at this node
		q, info_gain = self.find_best_question(x_train,y_train)

		if info_gain == 0: #if there is no point on asking questions
			label_occurs = DTree.get_label_occurences(y_train)
			return PNode(label_occurs) #node holds the probability of all possible predictions

		#partition data
		true_set, false_set = q.partition(x_train,y_train)

		true_node = self.fit(true_set["features"],true_set["labels"])
		false_node = self.fit(false_set["features"],false_set["labels"])

		question_node = QNode(q,true_node,false_node)
		self.root = question_node

		return question_node

	def predict(self,x_test):
		predicts = []
		for data_point in x_test:
			results = self.classify(data_point,self.root)
			predicts.append(max(results, key=results.get))

		return predicts

	#NOTE: either partition data and record how many test_points end up at each leaf, OR for-loop through all the test data
	def classify(self,data_point,curNode):

		#if we have reached a leaf
		if isinstance(curNode,PNode):
			return curNode.predicts #return list of predictions

		#otherwise keep partitioning the data
		direct = curNode.question.match(data_point)

		predict = None
		if direct == True:
			predict = self.classify(data_point,curNode.true_node)
		else:
			predict = self.classify(data_point,curNode.false_node)

		return predict

	def find_best_question(self,x_train,y_train):
		if len(x_train) == 0 or len(y_train) == 0: 
			return None

		#generate questions =-=-=-=-=
		#first find all unique features (this breaks if data is empty)
		unq_features = DTree.find_unique_features(x_train)
		
		cur_gini = DTree.gini(y_train) #Find current impurity
		best_gain = -1*sys.maxsize
		best_question = None

		#now iterate through the labels and find out what the best question to ask is
		for feature in unq_features:
			q = Question(unq_features[feature],feature)

			#Now partition the data and find the gini impurity and info gain
			true_set, false_set = q.partition(x_train,y_train)
			true_labels = true_set["labels"]
			false_labels = false_set["labels"]

			#Deterime a weighted gini value for this question
			new_gini = DTree.gini(true_labels)*(len(true_labels)/len(y_train)) + DTree.gini(false_labels)*(len(false_labels)/len(y_train))

			info_gain = cur_gini-new_gini
			if info_gain > best_gain:
				best_gain = info_gain
				best_question = q

		return best_question, best_gain

	def __repr__(self):
		return self.toString(self.root,0)

	def toString(self,curNode,recur_depth):
		if isinstance(curNode,PNode): #if we have reached a leaf
			return "\t"*recur_depth + str(curNode.predicts) + "\n"

		curString = "\t"*recur_depth + "Question: " + str(curNode.question) + "\n"
		if curNode.true_node != None:
			curString += ("True: " + self.toString(curNode.true_node,recur_depth+1))
		if curNode.false_node != None:
			curString += ("False: " + self.toString(curNode.false_node,recur_depth+1))

		return curString

	#helper methods
	@staticmethod
	def find_unique_features(features):
		unique = {} #stores the feature along with its column number (used to determine what type of feature it is)
		for datapoint in features:
			for i in range(len(datapoint)):
				feature = datapoint[i]
				if feature not in unique:
					unique[feature] = i
		return unique

	@staticmethod
	def get_label_occurences(y_train): #returns a dict with the amount of occurences each element has
		occurs = {}
		for label in y_train:
			if label in occurs:
				occurs[label] += 1
			else:
				occurs[label] = 1
		return occurs

	@staticmethod
	def gini(y_train): #input needs to be a 1D array of labels
		occurs = DTree.get_label_occurences(y_train)
		impurity = 1
		for label in occurs:
			prob = occurs[label] / len(y_train)
			impurity -= prob**2
		return impurity

class Question:

	def __init__(self,column,feature):
		self.column = column #the column number for the feature
		self.feature = feature #the specific feature

	def __repr__(self):
		return str(self.feature)

	def match(self,feature):
		if Question.is_numeric(self.feature): #if the feature is numeric, ask inequality questions
			return feature[self.column] <= self.feature
		else: #otherwise, if its a string, ask if they are equal
			return feature[self.column] == self.feature

	def partition(self,x_train,y_train): #takes in a dataset and partitinos it into true and false
		true_set = {"features" : [], "labels" : []}
		false_set = {"features" : [], "labels" : []}
		for i in range(len(x_train)):
			f = x_train[i]
			l = y_train[i]
			if self.match(f):
				true_set["features"].append(f)
				true_set["labels"].append(l)
			else:
				false_set["features"].append(f)
				false_set["labels"].append(l)
		return true_set, false_set

	#helper methods
	@staticmethod
	def is_numeric(val):
		return isinstance(val, int) or isinstance(val, float)

class QNode: #points towards child nodes and also holds a question

	def __init__(self,question,true_node,false_node):
		self.question = question
		self.true_node = true_node
		self.false_node = false_node

class PNode: #holds a list of possible labels

	def __init__(self,predict_set):
		self.predicts = predict_set

'''
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 4, 'Lemon'],
    ['Yellow', 3, 'Lemon'],
]

dtree = DTree()
dtree.fit(training_data)

print(dtree.predict([['Yellow', 3]]))

print(dtree)
'''

iris = load_iris()

#Split data into training and testing
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

#Create classifier
clf_tree = DTree()

#Train clasifiers
clf_tree.fit(x_train,y_train)

#Make predictions
tree_result = clf_tree.predict(x_test)

print(tree_result)
#Determine accuracy
print("tree accuracy:",accuracy_score(y_test,tree_result))
