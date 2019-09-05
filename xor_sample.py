import numpy as np
from nn import NN, Layer

X = np.array([
    [0,0,1],
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

print("Predictions before training")
print(nn.feed_forward(X))
print()

nn.train(X, Y)

print()
print("Predictions after training")
print(nn.feed_forward(X))