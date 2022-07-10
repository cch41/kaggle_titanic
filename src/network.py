import numpy as np
from clean_data import training_data, test_data

# the test file has no solution... has to be submitted to evaluate it.

# SGD on ReLu neurons


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x, y)
                        for y, x in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(shape=(y, 1)) for y in sizes[1:]]

    def SGD(self, training_data, lr, epochs, mini_batch_size):
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            np.random.shuffle(training_data)
            for mb_idx in range(0, len(training_data), mini_batch_size):
                print(f'mini batch')
                mini_batch = training_data[mb_idx:mb_idx+mini_batch_size]
                self.update_mini_batch(mini_batch, lr)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, lr):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for passenger, survival in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(passenger, survival)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w-(lr/len(mini_batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def feedforward_one(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        print(a)
        return a

    # def evaluate(self, eval_data):
    #     eval_input, eval_solution = eval_data
    #     eval_results = self.feedforward(eval_input)
    #     t = sum(x == y for x, y in zip(eval_results, eval_solution))
    #     return t

    def evaluate(self, eval_data):
        correct, total = 0, len(eval_data)
        for passenger, survival in eval_data:
            predicted_survival = self.feedforward_one(passenger)
            if predicted_survival == survival:
                correct += 1
        print(f"{(correct/total)*100}% out of {len(eval_data)}")


a = Network([6, 20, 20, 1])
a.SGD(training_data=training_data, lr=1,
      epochs=10, mini_batch_size=100)
