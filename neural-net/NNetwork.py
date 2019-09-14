import random as r

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

	def fit(self,x_train):
		for img in x_train:

			#Plug input matrix into input layer
			new_activation = matrix_to_array(img) 
			for i in range(len(new_activation)):
				self.network[0][i].activation = new_activation[i]

			#Activate next layers
			for l in range(len(self.network)-1): 

				#calculate resultant activation
				weight_matrix = np.array([w for w in self.network[l][n].synapsis.values()] for n in range(len(self.network[l])))
				neuron_vector = np.array([n for n in self.network[l].activation])
				bias_vector = np.array([n for b in self.network[l].bias])

				#the dot product of the weight matrix and the activation vector is equal to the weighted average vector
				weightedAvg_vector = weight_matrix.dot(neuron_vector) + bias_vector
				resultant_activation = matrix_to_array(softplus(weightedAvg_vector)) #the activation of the next layer

				#apply activatoin to next layer
				for i in range(len(resultant_activation)):
					self.network[l+1][i] = resultant_activation[i]

	#def predict(self,):

	def __repr__(self):
		string = "Network info =-=-=-=-=-=-=-"
		for l in range(len(self.network)):
			layer = self.network[l]
			string += ("\n layer="+str(l)+", neuron_count="+str(len(layer)))

		return string

	@staticmethod
	def matrix_to_array(self,matrix):
		return matrix.flatten().tolist()

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
