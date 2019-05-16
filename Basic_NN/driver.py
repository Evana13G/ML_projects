# Built from the guidance of: 
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
# https://www.python-course.eu/neural_networks_with_python_numpy.php

import numpy as np
from neural_network import NeuralNetwork

x = []
y = []
x.append([0,0,1])
x.append([0,1,1])
x.append([1,0,0])
x.append([1,1,1])
y.append([0])
y.append([1])
y.append([1])
y.append([0])

x_training = np.array(x)
y_training = np.array(y)
no_passes = 1500

print('\n' + "-----------------------------------------------------------")
print("--------------------- Neural Net Test ---------------------")
print("-----------------------------------------------------------" + '\n')
NN = NeuralNetwork(x_training, y_training)
NN.train(no_passes)
