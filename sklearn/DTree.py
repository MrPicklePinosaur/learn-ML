import numpy as np
from pprint import *

class DTree:

	def fit(self,dataset): #this is where we build the tree
		pass

	def predict(self):
		pass

	def ask_questions(self,dataset):
		#generate questions =-=-=-=-=
		#first find all unique features (this breaks if data is empty)
		labels = DTree.remove_labels(dataset) #truncate dataset so we only look at features
		unq_labels = DTree.find_unique_features(labels)

		#now iterate through the labels and find out what the best question to ask is
		for label in unq_labels:
			q = Question(unq_labels[label],label)
			print("Question is",label)

			#Now partition the data and find the gini impurity and info gain
			part = q.partition(dataset)
			true_set = part["true"]
			false_set = part["false"]
			print(true_set)
			print(false_set)

		

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

class Question:

	def __init__(self,column,feature):
		self.column = column #the column number for the feature
		self.feature = feature #the specific feature

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
dtree.ask_questions(training_data)

#pprint(np.delete(training_data,len(training_data[0])-1,axis=1))