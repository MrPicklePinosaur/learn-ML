import random as r
import numpy as np
import tensorflow as tf

from Neuron import *

class NNetwork:

	def __init__(self):
		self.network = [] #2d array, with each interior array being a layer in the network
		self.batch_size = 16 

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
					weight = r.randint(-100,100)/100 #init with random weight
					cur_neuron.connect_neuron(next_neuron,weight)

	def fit(self,x_train,y_train):
		#shuffle dataset`to employ stochastic descent
		rand = [i for i in range(len(x_train))]
		r.shuffle(rand)
		x_train = [x_train[n] for n in rand]
		y_train = [y_train[n] for n in rand]

		result = self.predict(x_train[0])

		w, b, a = self.backprop(self.network[-1][0],len(self.network)-1)

		#for i in range(len(x_train)):

		'''
		for l in range(len(self.network)-1,0,-1): #start from the output layer and work backwards
			layer = self.network[l]

			#calculate how the current test case would like to affect the weights/biases
			avg_ratio_vector = np.array([len(layer)])
			avg_bias_vector = np.array([len(layer)])
			for output_node in layer:
				ratio_vector, bias_vector = self.weight_backprop(output_node,l) #find components for gradient vector
				avg_ratio_vector = np.add(avg_ratio_vector,ratio_vector)
				bias_ratio_vector = np.add(avg_bias_vector,bias_vector)

			avg_ratio_vector /= len(layer)
			avg_bias_vector /= len(layer)

		'''

		''' cost function stuff
		#determine how good the current prediction is
		result = self.predict(x_train[i])
		expected = [1 if y_train[i] == i else 0 for i in range(10)] #convert expected output into an output layer list
		avg_cost = NNetwork.cost_function(result,expected)
		'''
	
	'''
	def backprop(self,root_neuron,layer_index): 
		ratio_vector = []
		bias_vector = []
		for prev_neuron in self.network[layer_index-1]: 

			#weight - Chain rule: the change in the cost function with respect to the activation of the node in the previous layer
			ratio = prev_neuron.activation*NNetwork.derv_softplus(prev_neuron.activation*prev_neuron.synapsis[root_neuron]+root_neuron.bias)*2*(root_neuron.activation-prev_neuron.synapsis[root_neuron])
			ratio_vector.append(ratio)

			#bias
			bias = NNetwork.derv_softplus(prev_neuron.activation*prev_neuron.synapsis[root_neuron]+root_neuron.bias)*2*(root_neuron.activation-prev_neuron.synapsis[root_neuron])
			bias_vector.append(bias)

		# vector with height n where n is the number of neuroins n previous layers
		return np.array([ratio_vector]), np.array([bias_vector])
	'''

	def backprop(self,cur_neuron,layer_index):
		if (layer_index-1 == 0): 
			#giant mess of calc
			for prev_neuron in self.network[layer_index-1]: #for each node in previous layer
				weight = prev_neuron.activation*NNetwork.derv_softplus(prev_neuron.activation*prev_neuron.synapsis[cur_neuron]+cur_neuron.bias)*2*(cur_neuron.activation-prev_neuron.synapsis[cur_neuron])
				bias = NNetwork.derv_softplus(prev_neuron.activation*prev_neuron.synapsis[cur_neuron]+cur_neuron.bias)*2*(cur_neuron.activation-prev_neuron.synapsis[cur_neuron])
				activation = prev_neuron.synapsis[cur_neuron]*NNetwork.derv_softplus(prev_neuron.activation*prev_neuron.synapsis[cur_neuron]+cur_neuron.bias)*2*(cur_neuron.activation-prev_neuron.synapsis[cur_neuron])

			return weight, bias, activation

		#merge previous results and bubble upwards
		ratio_vector = []
		bias_vector = []
		activation_vector = []
		for neuron in self.network[layer_index]: #for all nodes in current layer, match with all nodes in previous layer
			w, b, a = self.backprop(neuron,layer_index-1)
			ratio_vector.append(w)
			bias_vector.append(b)
			activation_vector.append(a)

		return ratio_vector, bias_vector, activation_vector


	''' TODO
	split train data into mini batches
	create gradient vector
	make it so the backprop algo can be modifyable
	'''

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
					self.network[l+1][i].activation = resultant_activation[i]

		return [n.activation for n in self.network[-1]]

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
		return 1/(1+np.exp(-x))

	@staticmethod
	def cost_function(result,expected): #used as a measure on how sucessful our optimization is
		assert len(result) == len(expected), "Unable to determine cost, array lengths are different"
		avg_cost = 0
		for i in range(len(result)):
			avg_cost += (result[i]-expected[i])**2
		return avg_cost/len(result)

mnist = tf.keras.datasets.mnist
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #set log type

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = x_train/255.0, y_train/255.0, x_test/255.0, y_test/255.0

net = NNetwork()
blueprint = [784,16,16,10]
net.build_network(blueprint)

net.fit(x_test,y_test)