import numpy as np 
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import os

# =================================================================================
# Forward propagation
# Define the relu activation function
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Example of forward propagation
input_data = np.array([3,5])
weights = {'node_0_0': np.array([2,4]),
            'node_0_1': np.array([4,-5]),
            'node_1_0': np.array([-1,2]),
            'node_1_1': np.array([1,2]),
            'output': np.array([2,7])}

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input) # Compute the activation function

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)


# =============================================================================
# Gradient descent
# How to calculate slopes and update weights

weights = np.array([1,2]) 
input_data = np.array([3,4])
target = 6
learning_rate = 0.01
preds = (input_data * weights).sum()
error = preds - target
print(error)

gradient = 2 * input_data * error
weights_update = weights - learning_rate*gradient
preds_updated = (input_data * weights_update).sum()
error_updated = preds_updated - target
print(error_updated)

# =============================================================================
# Creating a deep neural network with Keras
# 
mnist_train = pd.read_csv('/Users/licheng/Training/data/mnist_train.csv')
mnist_test = pd.read_csv('/Users/licheng/Training/data/mnist_test.csv')

predictors = mnist_train.loc[:,'1x1':'28x28'].to_numpy()
target = to_categorical(mnist_train['label'])

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors,target,validation_split = 0.3)














