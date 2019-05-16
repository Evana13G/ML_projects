import numpy as np
import math
from util import sigmoid, sigmoid_derivative

class NeuralNetwork: 
	def __init__(self, x, y):
		# self.training_input = x
		# self.training_output = y

		self.input = x #lets say its an array np.array([4, 3, 4, 5])
		self.weights1 = np.random.rand(self.input.shape[1], 4) # Tuple of the dimensions of an array 
										 # In our case, the x[1] is the rows of the input RXC 
										 # aka the height
										 # You choose how many layers you want the middle layers to have
		self.weights2 = np.random.rand(4,1) # Not sure why 
		self.output = y
		self.output_hat = np.zeros(self.output.shape)

	def train(self, cycles):
		for cycle in range(cycles):
			print('\n' + "----------------- Pass number " + str(cycle) + " -----------------" + '\n')
			self.feedforward()
			self.backprop()
			self.print_output()

	def feedforward(self):
		# Assume Bias is zero
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.output_hat = sigmoid(np.dot(self.layer1, self.weights2))


	### Backpropogating the error
	def backprop(self):
		# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
		# In the notes,    dLoss(y - y_hat)/dW 
		# Expands to ...   = dLoss(y - y_hat)/dy_hat * dy_hat/dz * dz/dW 
		#                              where z = Wx + b
		#                  = 2(y-y_hat) * deriv sigmoid * d(Wx + b)/dW
		#                  = 2(y-y_hat) * z(1-z) * x
		# 
		# NOTICE weight2 then weights1 are handled ... reverse order 
		d_weights2 = np.dot(self.layer1.T, (2*(self.output - self.output_hat) * sigmoid_derivative(self.output_hat)))
		d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.output - self.output_hat) * sigmoid_derivative(self.output_hat), self.weights2.T) * sigmoid_derivative(self.layer1)))

		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def loss_function(self):
		sum_sq = 0
		for i in range(len(self.output_hat)):
			diff = self.output[i] - self.output_hat[i]
			sum_sq = sum_sq + math.pow(diff, 2)
		return sum_sq

###################################################################################################

	def print_output(self):
		# print('\n' + "-----------------------------------------------------------")
		# print("--------------------- Neural Net Test ---------------------")
		# print("-----------------------------------------------------------" + '\n')
		
		print("-- INPUT values: " + str(self.input) + '\n')
		print("-- OUTPUT values: " + str(self.output) + '\n')

		print("-- Layer 1 values: " + str(self.layer1) + '\n')
		print("-- Output layer values: " + str(self.output_hat) + '\n')
		print("---- Total Loss: " + str(self.loss_function()))



