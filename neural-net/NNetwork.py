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
					weight = r.randint(0,100)/100 #generate with random weight between 0 and 1
					cur_neuron.connect_neuron(next_neuron,weight)

	def __repr__(self):
		string = "Network info =-=-=-=-=-=-=-"
		for l in range(len(self.network)):
			layer = self.network[l]
			string += ("\n layer="+str(l)+", neuron_count="+str(len(layer)))

		return string

net = NNetwork()
blueprint = [784,16,16,10]
net.build_network(blueprint)
print(net)
print(net.network)