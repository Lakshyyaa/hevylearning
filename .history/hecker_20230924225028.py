import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read dataset, source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
data = pd.read_csv('heart.csv')

# Remove output collumn from the dataset
X = data.drop('output', axis=1).values
y = data['output'].values.reshape(-1, 1)

# Normalizing/Scaling features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Designing the NN with single hidden layer
input_size = X.shape[1]
hidden_size = 10 
output_size = 1
learning_rate = 0.01
num_epochs = 1000

# Randomizing the weights and biases
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Output Function: Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cost Function: Binary cross entropy
def binarycross(output_layer_output, y):
    return -np.mean(y * np.log(output_layer_output) + (1 - y) * np.log(1 - output_layer_output))

costs = []

for epoch in range(num_epochs):
    # Forward propagation : On each layer we are doing aggreggation and activation and feeding forward to the
    # next layer
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    
    # Computing cost using the binary cross entropy cost function
    cost = binarycross(output_layer_output, y)
    costs.append(cost)
    
    # Backpropagation : Finding the derivates with respect to the parameters
    d_output = (output_layer_output - y) / len(y)
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
    
    # Updating weights & biases 
    weights_hidden_output -= hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden -= X.T.dot(d_hidden) * learning_rate
    bias_hidden -= np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Cost: {cost}')

# Plotting cost vs. iterations
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()
