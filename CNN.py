import numpy as np
import math



def convolutional_layer(input_matrix, size, filter_matrix, filter_measurement): #i would have just used np.convolve but i needed relu too
    output = np.zeros((size - 1, size - 1))
    steps = size - 1
    for i in range(steps):
        for j in range(steps):
            output[i][j] = max(0, np.sum(filter_matrix * input_matrix[i: i + filter_measurement, j: j + filter_measurement])) # relu here
    return output



def max_pooling(input_matrix, size, pool_measurement):
    output = np.zeros((size - 1, size - 1))
    steps = size - 1
    for i in range(steps):
        for j in range(steps):
            output[i][j] = (input_matrix[i: i + pool_measurement, j: j + pool_measurement]).max()
    return output



def forwards_propagation(input_values, hidden_weights, output_weights):
    hidden_layer = np.sum(hidden_weights * input_values, 1)
    for i in range(len(hidden_layer)):
        hidden_layer[i] = max(0, hidden_layer[i])
    output_layer = np.sum(output_weights * hidden_layer, 1) 
    return hidden_layer, np.exp(output_layer)/np.sum(np.exp(output_layer))

def loss_function(expected, actual):
    loss = -(math.log(actual[expected], 10)) # - expected log(actual) cross entropy
    return loss


def backpropagation(output_weights, actual_output, expected_output, hidden_output, hidden_weights, input_matrix):
    lr = 0.001
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected[expected_output] = 1
    output_delta = np.subtract(actual_output, expected)
    output_weights = output_weights - (lr * hidden_output * np.transpose([output_delta])) 
    hidden_delta = np.dot(np.transpose(output_weights), output_delta) * np.where(hidden_output > 0, 1, 0) 
    hidden_weights = hidden_weights - (lr * np.dot(hidden_delta, input_matrix)) 
    return output_weights, hidden_weights 

def training(data, output_weights, hidden_weights, label):
    filter_matrix = [[1, 1], [-1, -1]]
    mymatrix = convolutional_layer(data/255, 28, filter_matrix, 2)
    mymatrix = max_pooling(mymatrix, 27, 2)
    results = forwards_propagation(mymatrix, hidden_weights, output_weights)
    output_weights, hidden_weights = backpropagation(output_weights, results[1], label, results[0], hidden_weights, mymatrix)
    return results[1], output_weights, hidden_weights

def testing(data, output_weights, hidden_weights, label):
    filter_matrix = [[1, 1], [-1, -1]]
    mymatrix = convolutional_layer(data/255, 28, filter_matrix, 2)
    mymatrix = max_pooling(mymatrix, 27, 2)
    results = forwards_propagation(mymatrix, hidden_weights, output_weights)
    return results[1]




f = open("path to training images") # I used the csv version of simplicity
g = open("path to testing images")
train_data = []
train_label = []
counter = 0
hidden_weights = np.random.rand(26, 26)
output_weights = np.random.rand(10, 26)
epochs = 10 
samples = 10000
for line in f:
    counter += 1
    line = np.fromstring(line, dtype=float, sep=',')
    train_label.append(line[0])
    line = np.reshape(np.delete(line, 0), (28, 28))
    train_data.append(line)
    if counter == samples:
        break

for i in range(epochs):
    loss = 0
    correct_pred = 0
    for j in range(samples):
        results, output_weights, hidden_weights = training(np.array(train_data[j]), output_weights, hidden_weights, int(train_label[j]))
        loss += loss_function(int(train_label[j]), results)
        if np.argmax(results) == int(train_label[j]): correct_pred += 1 
    print("Epoch", i + 1, ": Average loss =", loss/samples, "| Accuracy =", 100 * correct_pred/samples, "%")

test_data = []
test_label = []
counter = 0
tests = 5000
for line in g:
    counter += 1
    line = np.fromstring(line, dtype=float, sep=',')
    test_label.append(line[0])
    line = np.reshape(np.delete(line, 0), (28, 28))
    test_data.append(line)
    if counter > tests:
        break
loss = 0
correct_pred = 0
for k in range(tests):
    results = testing(np.array(train_data[k]), output_weights, hidden_weights, int(train_label[k]))
    loss += loss_function(int(train_label[k]), results)
    if np.argmax(results) == int(train_label[k]): correct_pred += 1 
print("Testing: Average loss =", loss/tests, "| Accuracy =", 100 * correct_pred/tests, "%")
