import numpy as np
import math
import operator
import re
import pandas as pd
from random import random

class MLP(object):
    def __init__(self, num_inputs=10, hidden_layers=[3, 3], num_outputs=1):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create different layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # randomize weights for the layers
        weights = []
        print("Initial Weights:")
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights
        for j in range(100):
            x = np.random.ranf()
            print("Edge [{}]: {}".format(j,x))
        print("==="*20)

        #derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        #activation function
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        activations = inputs

        #activations for backpropagation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            #sigmoid activation function
            activations = self._sigmoid(net_inputs)

            #activations for backpropagation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, error):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layers
            activations = self.activations[i+1]

            # sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta to get a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            #activations for current layer
            current_activations = self.activations[i]

            # reshape activations
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropagate error
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_errors = 0

            # iterate through the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                self.gradient_descent(learning_rate)

                # keep track of the MSE
                sum_errors += self._mse(target, output)

            # Epochs and error rate
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")


    def gradient_descent(self, learningRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
        


    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _mse(self, target, output):
        return np.average((target - output) ** 2)


def read_file(filepath) -> list:
    ds = []        
    with open(filepath) as fp:
        for line in fp:                              
            comp = re.compile("\d+")
            dataList = comp.findall(line)                   
            row = list(map(int, dataList[1:]))       
            ds.append(row)
    return ds

def create_df(dataset : list):
    dataframe = pd.DataFrame(dataset, columns=['a','b','c','d','e','f','g','h','i','j','label'])    
    return dataframe


if __name__ == "__main__":
    print("~"*50)
    print("\nMLP Architecture: ANN\nEpochs: 100\n")
    #reading dataset file
    dataset = read_file("dataset.txt")
    df = create_df(dataset)
    

    #spliting training and holdout set
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    holdout = df[~msk]
    items = train.drop(labels="label",axis=1)
    targets = train.drop(~items,axis=1)
    inp = np.array(items)
    tar = np.array(targets)

    #Spliting holdout set
    hold_items = train.drop(labels="label",axis=1)
    hold_targets = train.drop(~items,axis=1)
    hold_items = np.array(hold_items)

#     # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(10, [10,10], 1)

#     # train network
    mlp.train(inp, tar, 100, 0.01)


#validation set
    val = read_file("validationSet.txt")
    vf = create_df(val)
    vitems = df.drop(labels="label",axis=1)
    vtargets = df.drop(~vitems,axis=1)
    vinp = np.array(vitems)
    vtar = np.array(vtargets)





