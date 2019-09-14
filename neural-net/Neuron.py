import random as r

class Neuron:

	def __init__(self,layer,index):
		self.layer = layer
		self.index = index
		self.activation = r.randint(0,100)/100 #init neuron with random activation
		self.bias = 0
		self.synapsis = {} #Stores the next neuron as the key, and the weight as the index

	def connect_neuron(self,neuron,weight):
		self.synapsis[neuron] = weight

	def set_activation(self,activation):
		assert (0 <= activation and activation <= 1), ("Invalid activation provided: "+str(activation))
		self.activation = activation

	def __repr__(self):
		return "Neuron "+str(self.layer)+" "+str(self.index)

'''
n = Neuron(0,0)
n.set_activation(0.01)
print(n.activation)
'''