import math
import random

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the MLP with random weights and biases.
        """
        self.WH = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.WO = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.biasH = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.biasO = [random.uniform(-1, 1) for _ in range(output_size)]

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid for backpropagation."""
        return x * (1 - x)

    def feedforward(self, inputs):
        """
        Perform the feedforward operation.
        """
        self.hidden_outputs = [
            self.sigmoid(sum(i * w for i, w in zip(inputs, weights)) + bias)
            for weights, bias in zip(self.WH, self.biasH)
        ]
        self.final_outputs = [
            self.sigmoid(sum(h * w for h, w in zip(self.hidden_outputs, weights)) + bias)
            for weights, bias in zip(self.WO, self.biasO)
        ]
        return self.final_outputs

    def backpropagate(self, inputs, targets, learning_rate):
        """
        Perform backpropagation and update weights and biases.
        """
        # Compute errors
        output_errors = [t - o for t, o in zip(targets, self.final_outputs)]
        output_deltas = [e * self.sigmoid_derivative(o) for e, o in zip(output_errors, self.final_outputs)]

        hidden_errors = [
            sum(w * delta for w, delta in zip(weights, output_deltas))
            for weights in zip(*self.WO)
        ]
        hidden_deltas = [e * self.sigmoid_derivative(h) for e, h in zip(hidden_errors, self.hidden_outputs)]

        # Update weights and biases for output layer
        for i, weights in enumerate(self.WO):
            for j in range(len(weights)):
                self.WO[i][j] += learning_rate * output_deltas[i] * self.hidden_outputs[j]
            self.biasO[i] += learning_rate * output_deltas[i]

        # Update weights and biases for hidden layer
        for i, weights in enumerate(self.WH):
            for j in range(len(weights)):
                self.WH[i][j] += learning_rate * hidden_deltas[i] * inputs[j]
            self.biasH[i] += learning_rate * hidden_deltas[i]