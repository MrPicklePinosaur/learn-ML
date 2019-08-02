import numpy as np
import sys
from pprint import *

class DTree:

	def __init__(self):
		self.root = None

	def fit(self,dataset): #this is where we build the tree
		
		#Find the best question to ask at this node
		q = self.find_best_question(dataset)

		#sorta inefficient but we are going to have to partition data again
		part = q.partition(dataset)
		true_set = part["true"]
		false_set = part["false"]

		#TODO: DONT DO len(true_set), instead, check for unique features
		if len(true_set) > 0: #if we havent reached a leaf
			q.true_node = self.fit(true_set)
		if len(false_set) > 0:
			q.false_node = self.fit(false_set)

		return q

	def predict(self):
		pass

	def find_best_question(self,dataset):
		if dataset == None or len(dataset) == 0: 
			return None

		#generate questions =-=-=-=-=
		#first find all unique features (this breaks if data is empty)
		labels = DTree.remove_labels(dataset) #truncate dataset so we only look at features
		unq_labels = DTree.find_unique_features(labels)
		
		cur_gini = DTree.gini(dataset) #Find current impurity
		best_gini = sys.maxsize
		best_question = None

		#now iterate through the labels and find out what the best question to ask is
		for label in unq_labels:
			q = Question(unq_labels[label],label)

			#Now partition the data and find the gini impurity and info gain
			part = q.partition(dataset)
			true_set = part["true"]
			false_set = part["false"]

			#Deterime a weighted gini value for this question
			new_gini = DTree.gini(true_set)*(len(true_set)/len(dataset)) + DTree.gini(false_set)*(len(false_set)/len(dataset))

			if new_gini < best_gini: #NOTE: this might be wrong, try using info gain instead
				best_gini = new_gini
				best_question = q

		return best_question

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
	def get_occurences(dataset): #returns a dict with the amount of occurences each element has
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
		occurs = DTree.get_occurences(dataset)
		impurity = 1
		for label in occurs:
			prob = occurs[label] / len(dataset)
			impurity -= prob**2
		return impurity

class Question:

	def __init__(self,column,feature):
		self.column = column #the column number for the feature
		self.feature = feature #the specific feature

		self.true_node = None
		self.false_node = None

	def __repr__(self):
		return self.feature

	def match(self,example):
		if Question.is_numeric(self.feature): #if the feature is numeric, ask inequality questions
			return example[self.column] <= self.feature
		else: #otherwise, if its a string, ask if they are equal
			return example[self.column] == self.feature

	def partition(self,data): #takes in a dataset and partitinos it into true and false
		data_div = {"true" : [], "false": []}
		for example in data:
			if self.match(example):
				data_div["true"].append(example)
			else:
				data_div["false"].append(example)
		return data_div

	#helper methods
	@staticmethod
	def is_numeric(val):
		return isinstance(val, int) or isinstance(val, float)

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

dtree = DTree()
dtree.fit(training_data)

#print(DTree.get_occurences(training_data))
#pprint(np.delete(training_data,len(training_data[0])-1,axis=1))