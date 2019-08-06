import numpy as np
import sys
from pprint import *

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DTree:

	def __init__(self):
		self.root = None

	def fit(self,dataset): #this is where we build the tree
		
		#Find the best question to ask at this node
		q, info_gain = self.find_best_question(dataset)

		if info_gain == 0: #if there is no point on asking questions
			label_occurs = DTree.get_label_occurences(dataset)
			return PNode(label_occurs) #node holds the probability of all possible predictions

		#partition data
		true_set, false_set = q.partition(dataset)
		#print("true set:",true_set,"false_set",false_set)
		true_node = self.fit(true_set)
		false_node = self.fit(false_set)

		question_node = QNode(q,true_node,false_node)
		self.root = question_node

		return question_node

	def predict(self,data_set): #works for one point for now
		predicts = []
		for data_point in data_set:
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

	def find_best_question(self,dataset):
		if dataset == None or len(dataset) == 0: 
			return None

		#generate questions =-=-=-=-=
		#first find all unique features (this breaks if data is empty)
		labels = DTree.remove_labels(dataset) #truncate dataset so we only look at features
		unq_labels = DTree.find_unique_features(labels)
		
		cur_gini = DTree.gini(dataset) #Find current impurity
		best_gain = -1*sys.maxsize
		best_question = None

		#now iterate through the labels and find out what the best question to ask is
		for label in unq_labels:
			q = Question(unq_labels[label],label)

			#Now partition the data and find the gini impurity and info gain
			true_set, false_set = q.partition(dataset)

			#Deterime a weighted gini value for this question
			new_gini = DTree.gini(true_set)*(len(true_set)/len(dataset)) + DTree.gini(false_set)*(len(false_set)/len(dataset))

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
	def remove_labels(dataset):
		new_dataset = []
		for datapoint in dataset:
			new_dataset.append(datapoint[:len(datapoint)-1])
		return new_dataset

	@staticmethod
	def get_label_occurences(dataset): #returns a dict with the amount of occurences each element has
		occurs = {}
		for row in dataset:
			label = row[-1] #the label is always the last item in row 
			if label in occurs:
				occurs[label] += 1
			else:
				occurs[label] = 1
		return occurs

	@staticmethod
	def gini(dataset): #input needs to be a 1D array of labels
		occurs = DTree.get_label_occurences(dataset)
		impurity = 1
		for label in occurs:
			prob = occurs[label] / len(dataset)
			impurity -= prob**2
		return impurity

class Question:

	def __init__(self,column,feature):
		self.column = column #the column number for the feature
		self.feature = feature #the specific feature

	def __repr__(self):
		return str(self.feature)

	def match(self,example):
		if Question.is_numeric(self.feature): #if the feature is numeric, ask inequality questions
			return example[self.column] <= self.feature
		else: #otherwise, if its a string, ask if they are equal
			return example[self.column] == self.feature

	def partition(self,data): #takes in a dataset and partitinos it into true and false
		true_set = []
		false_set = []
		for example in data:
			if self.match(example):
				true_set.append(example)
			else:
				false_set.append(example)
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

#doing this cuz im too lazy to rewrite all the code to accomaddate 2 vars
train_set = []
for i in range(len(x_train)):
	row = []
	for n in x_train[i]:
		row.append(n)
	row.append(y_train[i])
	train_set.append(row)


#Create classifier
clf_tree = DTree()

#Train clasifiers
clf_tree.fit(train_set)

#Make predictions
tree_result = clf_tree.predict(x_test)

print(tree_result)
#Determine accuracy
print("tree accuracy:",accuracy_score(y_test,tree_result))
