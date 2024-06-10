import numpy as np
from PIL import Image


def activation_fun(value):
    return 1 / (1 + np.exp(-value))


def activation_derivative(value):
    return value * (1 - value)


# Function to load and preprocess images
# NN structure
# Layers
input_neurons_layer = 100 * 100  # For a 100X100 matrix
hidden_neurons_layer = 100  # Amount of neurons in the hidden layer
output_neurons_layer = 3  # 3 because we have 3 different shapes (Triangle, Circle and Eclipse)
# Weights
weights_from_input_to_hidden = np.random.uniform(
    size=(input_neurons_layer, hidden_neurons_layer))  # The weight from the input layer to the hidden layer
weights_from_hidden_to_output = np.random.uniform(
    size=(hidden_neurons_layer, output_neurons_layer))  # The weight from the hidden layer to the output layer
# Bias
bias_hidden = np.random.uniform(size=(1, hidden_neurons_layer))  # Ensure the shape matches the hidden layer
bias_output = np.random.uniform(size=(1, output_neurons_layer))  # Ensure the shape matches the output layer


# Feedforward process
# The function processes the entire input data in one go through matrix multiplications and vectorized operations,
# applying the neural network's weights and biases and then the activation function to produce the network's output efficiently.
# In this case the input is a vector of size 100*100 that represent an Image in black and white of a shape
# Where 1 is a black pixel and 0 is a white pixel.

def feedforward(input_Data):
    hidden_layer_activation = np.dot(input_Data,
                                     weights_from_input_to_hidden) + bias_hidden  #
    hidden_layer_output = activation_fun(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_from_hidden_to_output) + bias_output
    predicted_output = activation_fun(output_layer_activation)

    return hidden_layer_output, predicted_output  # The function is returning the expected output for a given input


# Backpropagation
def backpropagate(input_Data, hidden_layer_output, predicted_output,
                  actual_Output):  # responsible for updating the weights and biases of the network based on the errors computed during forward pass.
    global weights_from_input_to_hidden, weights_from_hidden_to_output
    global bias_hidden, bias_output

    # Calculate error
    error = actual_Output - predicted_output
    d_predicted_output = error * activation_derivative(predicted_output)

    # Hidden layer error
    error_hidden_layer = d_predicted_output.dot(weights_from_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * activation_derivative(
        hidden_layer_output)  # calculates the gradients of the loss function

    # Update weights and biases
    weights_from_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_from_input_to_hidden += input_Data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Training the neural network
# Epoch is One complete pass through the entire training dataset.
# By iterating through multiple epochs, the neural network progressively adjusts its weights to minimize the loss function,
# ideally improving its performance on the training data over time.
#
def train(input_Data, Actual_output, Epochs, learning_rate):
    for epoch in range(Epochs):
        hidden_layer_output, predicted_output = feedforward(
            input_Data)  # The function is returning the expected output for a given input foe the hidden layer and for the final output
        backpropagate(input_Data, hidden_layer_output, predicted_output, Actual_output)

        if epoch % 100 == 0:  # Printing the loss every 100 epochs helps monitor the training progress and check if the network is learning effectively.
            loss = np.mean(np.square(Actual_output - predicted_output))
            print(f'Epoch {epoch} Loss: {loss}')


# Example training data
input_data = np.random.rand(1000, input_neurons_layer)  # 1000 examples of 100x100 matrices
actual_output = np.random.randint(2, size=(1000, 1))  # Binary output (0 or 1)

# Training parameters
learning_rate = 0.01
epochs = 1000
