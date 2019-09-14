import random as r
import numpy as np

from Neuron import *

class NNetwork:

	def __init__(self):
		self.network = [] #2d array, with each interior array being a layer in the network

	#input is a list of ints, where each item in array represents the amount of neurons in that layer
	def build_network(self,blueprint):

		#create neurons in each layer of the network
		for l in range(len(blueprint)): #for number of layers
			layer = []
			for i in range(blueprint[l]): #for number of neurons in layer
				neuron = Neuron(l,i)
				layer.append(neuron)

			self.network.append(layer)

		#connect up the neurons with random weights
		for l in range(len(self.network)-1): #for each layer besides the last one
			for cur_neuron in self.network[l]:
				for next_neuron in self.network[l+1]:
					weight = r.randint(0,100)/100 #init with random weight
					cur_neuron.connect_neuron(next_neuron,weight)

	def fit(self,x_train,y_train):
		for i in range(len(x_train)):

			result = self.apply_input(x_train[i])
			expected = [1 if y_train[i] == i else 0 for i in range(10)] #convert expected output into an output layer list
			avg_cost = cost_function(result,expected)
			

	def predict(self,digits):
		for img in digits:

			#Plug input matrix into input layer
			new_activation = NNetwork.matrix_to_array(img) 

			for i in range(len(new_activation)):
				self.network[0][i].activation = new_activation[i]

			#Activate next layers
			for l in range(len(self.network)-1): 

				#calculate resultant activation
				weight_matrix = np.array(list([w for w in self.network[l][n].synapsis.values()] for n in range(len(self.network[l])))).transpose()
				neuron_vector = np.array([[n.activation for n in self.network[l]]]).transpose()
				bias_vector = np.array([[b.bias for b in self.network[l]]]).transpose()

				#the dot product of the weight matrix and the activation vector is equal to the weighted average vector
				weightedAvg_vector = np.dot(weight_matrix,neuron_vector)

				#np.add(np.dot(weight_matrix,neuron_vector),bias_vector)
				resultant_activation = NNetwork.matrix_to_array(NNetwork.softplus(weightedAvg_vector)) #the activation of the next layer

				#apply activatoin to next layer
				for i in range(len(resultant_activation)):
					self.network[l+1][i] = resultant_activation[i]

		return self.network[-1]

	def __repr__(self):
		string = "Network info =-=-=-=-=-=-=-"
		for l in range(len(self.network)):
			layer = self.network[l]
			string += ("\n layer="+str(l)+", neuron_count="+str(len(layer)))

		return string

	@staticmethod
	def matrix_to_array(matrix):
		return matrix.flatten().tolist()

	# 'squishification' functions
	@staticmethod
	def reLu(x):
		return max(0,x)

	@staticmethod
	def derv_reLu(x):
		if x < 0:
			return 0
		if x > 0:
			return 1
			
		return 0.5 #can be 0, 0.5 or 1, more testing needed

	@staticmethod
	def softplus(x):
		return np.log(1+np.exp(x))

	@staticmethod
	def derv_softplus(x): #the derivative of softplus happens to be sigmoid, neat!
		return 1/(1+np.log(-x))

	@staticmethod
	def cost_function(result,expected):
		assert len(result) == len(expected), "Unable to determine cost, array lengths are different"
		avg_cost = 0
		for i in range(len(result)):
			avg_cost += (results[i]-expected[i])**2
		return avg_cost/len(results)

