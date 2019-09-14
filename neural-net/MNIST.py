import tensorflow as tf
import numpy as np

from pprint import *
from NNetwork import *

mnist = tf.keras.datasets.mnist
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #set log type

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#pprint(x_test[0])


net = NNetwork()
blueprint = [784,16,16,10]
net.build_network(blueprint)
#print(net)
#print(net.network)

'''
weight_matrix = [[s.weight for s in net.network[0][n].synapsis.values()] for n in range(len(net.network[0]))]
print(len(weight_matrix))

neuron_matrix = [n.activation for n in net.network[0]]
print(len(neuron_matrix))
'''
bias_vector = [[b.bias for b in net.network[0][n].synapsis.values()] for n in range(len(net.network[0]))]
print(bias_vector[0])