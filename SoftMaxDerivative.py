
import numpy as np
import pandas as pd
import sympy as sp


def softmax(X):
    X = np.array(X)
    exp_X = np.exp(X - np.max(X))  # Numerical stability trick
    return exp_X / exp_X.sum()


def softmax_derivative(X):
    X = np.array(X)
    sftmx = softmax(X)
    matrix = np.zeros((len(X), len(X)))

    for r in range(len(X)):
        for c in range(len(X)):
            if r == c:
                matrix[r][c] = sftmx[r] * (1 - sftmx[r])
            else:
                matrix[r][c] = -sftmx[r] * sftmx[c]

    return sftmx, pd.DataFrame(matrix)


def cross_entropy_loss(Y, target):
    # Prevent log(0)
    epsilon = 1e-15
    Y = np.clip(Y, epsilon, 1 - epsilon)
    return -np.sum(target * np.log(Y))



# Example scenario
def demonstrate_softmax_cross_entropy():
    # Raw logits
    logits = np.array([1, 2, 3])

    # True target (one-hot encoded)
    target = np.array([0, 0, 1])

    # Softmax probabilities
    probabilities = softmax(logits)

    # Softmax derivative
    _, derivative_matrix = softmax_derivative(logits)

    # Cross-entropy loss
    loss = cross_entropy_loss(probabilities, target)

    print("Raw Logits:", logits)
    print("\nSoftmax Probabilities:", probabilities)
    print("\nTarget Probabilities:", target)
    print("\nDerivative Matrix:")
    print(derivative_matrix)

    # Demonstrating Loss Gradient
    loss_gradient = probabilities - target
    print("\nLoss Gradient (Y - target):", loss_gradient)

    # Showing how derivative matrix relates
    print("\nNote how gradient relates to softmax properties:")
    print("1. Highest probability (class 2) has largest negative adjustment")
    print("2. Other classes get pulled up to compensate")
    print("3. Gradient sum is zero, maintaining probability conservation")



demonstrate_softmax_cross_entropy()