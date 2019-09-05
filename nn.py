import numpy as np

#TODO add bias

LEARNING_RATE = 1
TRAINING_STEPS = 10000
LOG_PERIODS = 10

class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 2 * np.random.random((number_of_inputs, number_of_neurons)) - 1

    def backpropagate(self, error, learning_rate):
        e = error * self.nonlin(self.output, derivative=True)
        dCost = self.x.T.dot(e)
        self.weights -= learning_rate * dCost
        return e.dot(self.weights.T)
    
    def feedforward(self, x):
        self.x = x
        self.z = np.dot(self.x, self.weights)
        self.output = self.nonlin(self.z)
        return self.output
    
    def nonlin(self, x, derivative=False):
        if(derivative==True):
            return (x*(1-x))
        
        return 1/(1+np.exp(-x))
                
class NN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

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

    def mean_squared_error(self, yHat, y):
        return 2 * (yHat - y)

    def step(self, X, Y, learning_rate):
        yHat = self.feed_forward(X)
        error = self.mean_squared_error(yHat, Y)
        return self.backpropagate(error, learning_rate)
    
    def train(self, X, Y, learning_rate = LEARNING_RATE, steps = TRAINING_STEPS, log_periods = LOG_PERIODS):
        print("Training...")

        steps_per_periods = int(steps/log_periods)

        print("MSE:")
        for period in range(log_periods):
            for _ in range(steps_per_periods):
                error = self.step(X, Y, learning_rate)
            print("  Period", period, ":", np.mean(np.abs(error)))
