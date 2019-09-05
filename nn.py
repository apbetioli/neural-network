import numpy as np

#TODO add bias

class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 2 * np.random.random((number_of_inputs, number_of_neurons)) - 1

    def backpropagate(self, error, learning_rate):
        e = error * self.nonlin(self.activation, derivative=True)
        dCost = self.x.T.dot(e)
        self.weights -= learning_rate * dCost
        return e.dot(self.weights.T)
    
    def feedforward(self, x):
        self.x = x
        self.z = np.dot(self.x, self.weights)
        self.activation = self.nonlin(self.z)
        return self.activation
    
    def nonlin(self, x, derivative=False):
        if(derivative==True):
            return (x*(1-x))
        
        return 1/(1+np.exp(-x))
                
class NN:
    def __init__(self, learning_rate = 1):
        self.layers = []
        self.learning_rate = learning_rate

    def add(self, layer):
        self.layers.append(layer)
    
    def feed_forward(self, x):
        yHat = x
        for layer in self.layers:
            yHat = layer.feedforward(yHat)
        return yHat
        
    def train(self, X, Y):
        yHat = self.feed_forward(X)
        error = self.mean_squared_error(yHat, Y, derivative=True)
        self.backpropagate(error)
        return error
    
    def backpropagate(self, error):
        e0 = error
        for layer in reversed(self.layers):
            e0 = layer.backpropagate(e0, self.learning_rate)
    
    def mean_squared_error(self, yHat, y, derivative=False):
        return yHat - y if derivative else (yHat - y)**2


X = np.array([[0,0,1],
     [0,1,1],
     [1,0,1],
     [1,1,1]])

Y = np.array([[0],
             [1],
             [1],
             [0]])

   
nn = NN()
nn.add(Layer(3,3))
nn.add(Layer(3,1))

for j in range(60000):
    error = nn.train(X, Y)
    if(j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(error))))

nn.feed_forward(X)


