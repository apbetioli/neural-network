import numpy as np

class Layer:
    def __init__(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons

    def set_number_of_inputs(self, number_of_inputs):
        self.weights = 2 * np.random.random((number_of_inputs, self.number_of_neurons)) - 1
        self.bias = 2 * np.random.random((1, self.number_of_neurons)) - 1

    def backpropagate(self, error, learning_rate):
        e = error * self.nonlin(self.output, derivative=True)

        d_cost = self.x.T.dot(e)
        self.weights -= learning_rate * d_cost
        
        d_cost_bias = np.ones(len(self.x)).dot(e)
        self.bias -= learning_rate * d_cost_bias

        return e.dot(self.weights.T)
    
    def feedforward(self, x):
        self.x = x
        self.z = np.dot(self.x, self.weights) + self.bias
        self.output = self.nonlin(self.z)
        return self.output
    
    def nonlin(self, x, derivative=False):
        if(derivative==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
                
class NN:
    def __init__(self, number_of_inputs):
        self.layers = []
        self.next_number_of_inputs = number_of_inputs

    def add(self, layer):
        layer.set_number_of_inputs(self.next_number_of_inputs)
        self.layers.append(layer)
        self.next_number_of_inputs = layer.number_of_neurons

    def backpropagate(self, error, learning_rate):
        e = error
        for layer in reversed(self.layers):
            e = layer.backpropagate(e, learning_rate)
        return e
    
    def feed_forward(self, x):
        yHat = x
        for layer in self.layers:
            yHat = layer.feedforward(yHat)
        return yHat

    def mean_squared_error(self, yHat, y, derivative=True):
        return yHat - y if derivative else 0.5 * (yHat - y) ** 2

    def step(self, X, Y, learning_rate):
        yHat = self.feed_forward(X)
        error = self.mean_squared_error(yHat, Y)
        return self.backpropagate(error, learning_rate)
    
    def train(self, X, Y, learning_rate = 1, steps = 10000, log_periods = 10):
        print("Training...")

        steps_per_periods = int(steps/log_periods)

        print("MSE:")
        for period in range(log_periods):
            for _ in range(steps_per_periods):
                error = self.step(X, Y, learning_rate)
            print("  Period", period, ":", np.mean(np.abs(error)))
