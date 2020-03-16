import numpy as np
import pandas as pd
import csv

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        global numFactors
        numFactors = 4
        self.synaptic_weights = 2 * np.random.random((numFactors,1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)
    
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

def main():
    #setup
    print()
    file = "salaryData.csv"
    neural_network = NeuralNetwork()
    all_data = np.genfromtxt(file, delimiter=",")
    print("all data", all_data)
    #training inputs
    scaleInputs = all_data[1]
    scaleInputs = np.delete(scaleInputs, numFactors)
    print("scales\n", scaleInputs, end="\n\n")
    factors = all_data[0]
    factors = np.delete(factors, numFactors)
    print("training inputs normal\n", factors)
    
    training_inputs = all_data[2:] #skip header and scales
    training_inputs = np.delete(training_inputs, 4, axis=1)
    
                        
    print(training_inputs, end="\n\n")
    training_inputs /= scaleInputs
    print("training inputs scaled to 0-1\n", factors)
    print(training_inputs, end="\n\n")

    #training outputs
    result = all_data[0][len(all_data[0]) - 1]
    scaleOutput = 1.0 / all_data[1][numFactors]
    print("scaleoutput", scaleOutput)

    given_outputs = [all_data[2][numFactors]]
    for row in range(3, len(all_data)):
        given_outputs.append(all_data[row][numFactors])
    given_outputs = [given_outputs]
    print("given outputs", given_outputs)
    training_outputs = (np.multiply(scaleOutput, given_outputs)).T
    print("training outputs scaled")
    print(training_outputs, end="\n\n")


    #train
    neural_network.train(training_inputs, training_outputs, 100000)

    
    
    #show
    print("synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = float(input(factors[0]))
    B = float(input(factors[1]))
    C = float(input(factors[2]))
    D = float(input(factors[3]))

    print("new situation input data", A, B, C, D)
    print(result, " data")
    new_inputs = np.array([A,B,C,D])
    print(neural_network.think(new_inputs / scaleInputs) / scaleOutput)

main() 
