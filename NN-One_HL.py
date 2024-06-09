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
hidden_neurons_layer = 10  # Amount of neurons in the hidden layer
output_neurons_layer = 3  # 3 because we have 3 different shapes (triangle, circle and eclipse)
# Weights
weights_input_hidden = np.random.uniform(size=(input_neurons_layer, hidden_neurons_layer))
weights_hidden_output = np.random.uniform(size=(hidden_neurons_layer, output_neurons_layer))
# Bias
bias_hidden = np.random.uniform(size=(1, hidden_neurons_layer))  # Ensure the shape matches the hidden layer
bias_output = np.random.uniform(size=(1, output_neurons_layer))  # Ensure the shape matches the output layer


# Feedforward process
# computes the output of the network given an input.
def feedforward(input_Data):
    hidden_layer_activation = np.dot(input_Data,
                                     weights_input_hidden) + bias_hidden  # np.dot can handle multiplications in different dimensions.
    hidden_layer_output = activation_fun(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = activation_fun(output_layer_activation)

    return hidden_layer_output, predicted_output


# Backpropagation
def backpropagate(input_Data, hidden_layer_output, predicted_output,
                  actual_output):  # responsible for updating the weights and biases of the network based on the errors computed during forward pass.
    global weights_input_hidden, weights_hidden_output
    global bias_hidden, bias_output

    # Calculate error
    error = actual_output - predicted_output
    d_predicted_output = error * activation_derivative(predicted_output)

    # Hidden layer error
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * activation_derivative(
        hidden_layer_output)  # calculates the gradients of the loss function

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += input_Data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Training the neural network
def train(input_Data, Actual_output, Epochs, learning_rate):
    for epoch in range(Epochs):
        hidden_layer_output, predicted_output = feedforward(input_Data)
        backpropagate(input_Data, hidden_layer_output, predicted_output, Actual_output)

        if epoch % 100 == 0:
            loss = np.mean(np.square(Actual_output - predicted_output))
            print(f'Epoch {epoch} Loss: {loss}')


# Example training data
# Note: This is placeholder data. You need actual labeled training data for this task.
input_data = np.random.rand(1000, input_neurons_layer)  # 1000 examples of 100x100 matrices
actual_output = np.random.randint(2, size=(1000, 1))  # Binary output (0 or 1)

# Training parameters
learning_rate = 0.01
epochs = 1000

