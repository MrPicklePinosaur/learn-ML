import numpy as np
from pprint import *

class DTree():

	def fit(self,dataset): #this is where we build the tree
		pass

	def predict(self):
		pass

	def ask_questions(self,dataset):
		#generate questions =-=-=-=-=
		#first find all unique features (this breaks if data is empty)
		dataset = np.delete(dataset,len(dataset[0])-1,axis=1) #truncate dataset so we only look at features
		unq_labels = find_unique_features(dataset)

		#now iterate through the labels and ask questions
		

	#helper methods
	@staticmethod
	def find_unique_features(features):
		unique = []
		for datapoint in features:
			for feature in datapoint:
				if feature not in unique:
					unique.append(feature)
		return unique

class DNode():

	def __init__(self,column,row):
		self.column = column #the column number for the feature
		self.row = row #all the data for a certain item

	def match(self,example):
		if is_numeric(self.feature): #if the feature is numeric, ask inequality questions
			return example[self.column] <= self.row[self.column]
		else: #otherwise, if its a string, ask if they are equal
			return example[self.column] == self.row[self.column]

	#helper methods
	@staticmethod
	def is_numeric(val):
		isinstance(value, int) or isinstance(value, float)

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

#pprint(np.delete(training_data,len(training_data[0])-1,axis=1))
#q = DNode()