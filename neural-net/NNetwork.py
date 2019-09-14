import random as r

from Neuron import *
from Synapsis import *

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
					synapsis = Synapsis() #initialized with random weight and bias
					cur_neuron.connect_neuron(next_neuron,synapsis)

	def fit(self,x_train):
		for l in range(len(self.network)-1):
			weight_matrix = np.array([s.weight for s in self.network[l][n].synapsis.values()] for n in range(len(self.network[l])))
			neuron_vector = np.array([n for n in self.network[l].activation])
			bias_vector = np.array([b.bias for b in self.network[l][n].synapsis.values()] for n in range(len(self.network[l])))

		#the dot product of the weight matrix and the activation vector is equal to the weighted average vector
		weightedAvg_vector = weight_matrix.dot(neuron_vector)
		resultantNeuron_vector = weightedAvg_vector + bias_vector

	#def predict(self,):


	def __repr__(self):
		string = "Network info =-=-=-=-=-=-=-"
		for l in range(len(self.network)):
			layer = self.network[l]
			string += ("\n layer="+str(l)+", neuron_count="+str(len(layer)))

		return string

	# 'squishification' functions
	@staticmethod
	def reLu(self,x):
		return max(0,x)

	@staticmethod
	def derv_reLu(self,x):
		if x < 0:
			return 0
		if x > 0:
			return 1
			
		return 0.5 #can be 0, 0.5 or 1, more testing needed

	@staticmethod
	def softplus(self,x):
		return np.log(1+np.exp(x))

	@staticmethod
	def derv_softplus(self,x): #the derivative of softplus happens to be sigmoid, neat!
		return 1/(1+np.log(-x))
