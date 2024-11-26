import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

green_print = lambda x: f"\033[30;42m{x}\033[0m"
red_print = lambda x: f"\033[30;41m{x}\033[0m"
orange_print = lambda x: f"\033[1;33;40m{x}\033[0m"
yellow_print = lambda x: f"\033[30;43m{x}\033[0m"
blue_print = lambda x: f"\033[30;44m{x}\033[0m"
magenta_print = lambda x: f"\033[30;45m{x}\033[0m"
cyan_print = lambda x: f"\033[30;46m{x}\033[0m"


def softmax(Z):
    """
    Applies the softmax function to convert raw scores to probabilities.
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

    :param Z: 2D array of shape (batch_size, num_classes)
    :return: 2D array of same shape with softmax applied row-wise
    """
    # Subtract max for numerical stability (prevents overflow)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def ReLU(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)

    :param Z: Input array
    :return: Array with ReLU applied element-wise
    """
    return np.maximum(0, Z)


def deriv_ReLU(Z):
    """
    Computes the derivative of the ReLU function.
    The derivative is 1 for x > 0, and 0 for x <= 0.

    :param Z: Input array
    :return: Array with derivative of ReLU applied element-wise
    """
    return Z > 0  # Returns a boolean array, which is implicitly converted to 0s and 1s


class SimpleGradient:
    def __init__(self, x_size=8, y_size=4):
        self.X = np.random.uniform(-100, 100, size=x_size)
        self.X = (self.X - self.X.mean()) / self.X.std()

        self.W1 = np.random.randn(y_size, x_size) * 0.01  # Random small weights
        self.A1 = np.maximum(self.W1 @ self.X, 0)  # ReLU

        self.W2 = np.random.randn(y_size, y_size) * 0.01  # Square matrix
        self.A2 = self.W2 @ self.A1

        self.Y = softmax(self.A2.reshape(1, -1))

        self.target = np.zeros(self.Y.size)
        self.target[random.randint(0, len(self.target) - 1)] = 1

        self.loss = np.sum((self.Y - self.target) ** 2)

        """
                    we want dLoss/dW2 & dLoss/dW1

                    dLoss/dW2 = dLoss/dA2 * dA2/dW2

                    dLoss/dW1 = dLoss/dA2 * dA2/dW2 * dW2/dA1 * dA1/dW1 

        """

        print(
            f"\n\nINITIALIZATION:"
            f"\n************************************************************************************\n"
            f"{blue_print("X")}: {pd.DataFrame(self.X).T}, "
            f"\n{orange_print("W1")}: {pd.DataFrame(self.W1)}, "
            f"\n{cyan_print("A1")}: {pd.DataFrame(self.A1)}, "
            f"\n{magenta_print("W2")}: {pd.DataFrame(self.W2)}, "
            f"\n{cyan_print("A2")}: {pd.DataFrame(self.A2)}, "
            f"\n{yellow_print("Output")}: {pd.DataFrame(self.Y).T}, "
            f"\n{green_print("Target")}: {pd.DataFrame(self.target)}, "
            f"\n{red_print("Loss")}: {self.loss}"
            f"\n************************************************************************************")

        self.losses = self.weight1_trajectories = self.weight2_trajectories = self.output_trajectories = None

    def gradient_descent(self, iterations=100, learning_rate=0.001, plot=True):
        self.losses = [self.loss]
        self.weight1_trajectories = np.zeros((self.W1.shape[0], self.W1.shape[1], iterations))
        self.weight1_trajectories = np.zeros((self.W2.shape[0], self.W2.shape[1], iterations))
        self.output_trajectories = np.zeros((self.Y.size, iterations))

        self.weight1_trajectories[:, :, 0] = self.W1
        self.weight2_trajectories[:, :, 0] = self.W2
        self.output_trajectories[:, 0] = self.Y

        for iteration in range(1, iterations):
            """
            we want dLoss/dW2 & dLoss/dW1

            dLoss/dW2 = dLoss/dA2 * dA2/dW2

            dLoss/dW1 = dLoss/dA2 * dA2/dW2 * dW2/dA1 * dA1/dW1 





                    # Backpropagation
dLoss_dA2 = 2 * (Y - target)  # Loss derivative w.r.t. A2
dA2_dW2 = A1                  # Gradient of A2 w.r.t. W2

dLoss_dW2 = dLoss_dA2.T @ dA2_dW2.reshape(1, -1)

dA2_dA1 = W2.T
dA1_dW1 = relu_derivative(A1)[:, np.newaxis] * X[np.newaxis, :]  # Chain rule with ReLU

dLoss_dW1 = (dLoss_dA2 @ dA2_dA1) * dA1_dW1
            """

            dLoss_dA2 = 2 * (self.Y - self.target)  # Loss derivative w.r.t. A2
            dA2_dW2 = self.A1  # Gradient of A2 w.r.t. W2

            # Gradient computation
            dloss_dY = 2 * (self.Y - self.target)  # Gradient wrt outputs

            dloss_dW1 = np.outer(dloss_dY, self.X)  # Gradient wrt weights
            if iteration % 25 == 0:
                print(
                    f"\n\n------------------------------------------------------------------------------------------------")
                print(f"Iteration {iteration}:")
                print(
                    f"{blue_print("X")}: {pd.DataFrame(self.X).T}, "
                    f"\n{yellow_print("Output")}: {pd.DataFrame(self.Y).T}, "
                    f"\n{green_print("Target")}: {pd.DataFrame(self.target).T}, "
                    f"\n{red_print("Loss")}: {self.loss}")
                print(
                    f"\n\t{cyan_print("Gradient wrt Outputs")}: (d-{red_print("Loss")} / d-{yellow_print("Output")}): \n\n\t\t{pd.DataFrame(dloss_dY).T}"
                    f"\n\n\tFormula: (2 * ({yellow_print("Output")} - {green_print("Target")}))")
                print(
                    f"\n\n\t{magenta_print("Gradient wrt Weights")}: (d-({cyan_print("Gradient wrt Outputs")}) / d-{blue_print("X")}): \n{pd.DataFrame(dloss_dW1)}"
                    f"\n\n\tFormula: (OuterProduct ({cyan_print("Gradient wrt Outputs")}) x {blue_print("X")})")
                print(f"Now we just subtract this ^^^ matrix (* learning_rate={learning_rate}) from our previous W1")
                print(
                    f"------------------------------------------------------------------------------------------------")

            # Update weights
            self.W1 -= learning_rate * dloss_dW1
            self.weight1_trajectories[:, :, iteration] = self.W1  # Store updated weights

            self.W2 -= learning_rate * dloss_dW2
            self.weight2_trajectories[:, :, iteration] = self.W2  # Store updated weights

            # Recompute predictions and loss
            self.A1 = np.maximum(self.W1 @ self.X, 0)  # ReLU
            self.A2 = self.W2 @ self.A1
            self.Y = softmax(self.A2.reshape(1, -1))

            self.output_trajectories[:, iteration] = self.Y  # Store updated outputs
            self.loss = sum((self.Y - self.target) ** 2)
            self.losses.append(self.loss)

        if plot:
            self._plot_gd(learning_rate)

    def _plot_gd(self, lr):
        fig = plt.figure(figsize=(16, 12))  # Slightly taller to accommodate the bar chart
        gs = GridSpec(3, 2, width_ratios=[2, 3], height_ratios=[0.5, 1, 1], figure=fig)

        # Use a color palette that matches the outputs over time graph
        colors = plt.cm.tab10(np.linspace(0, 1, self.Y.size))

        # Target Output Bar Chart (new top subplot)
        ax_target = fig.add_subplot(gs[0, :])

        # Create bar plot with value labels
        bars = ax_target.bar(
            range(len(self.target)),
            self.target,
            color=colors,
            alpha=0.7
        )

        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax_target.text(
                bar.get_x() + bar.get_width() / 2.,
                0,  # Position at x-axis instead of bar top
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='gray',
                rotation=0,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
            )

        ax_target.set_title("Target Outputs")
        ax_target.set_xlabel("Output Index")
        ax_target.set_ylabel("Target Value")
        ax_target.set_xticks(range(len(self.target)))
        ax_target.set_xticklabels([f'[{i}]' for i in range(len(self.target))])

        # Outputs plot
        ax_outputs = fig.add_subplot(gs[1, 0])
        final_outputs = []
        for i in range(self.Y.size):
            outputs = self.output_trajectories[i, :]
            final_outputs.append(outputs[-1])
            ax_outputs.plot(
                outputs,
                label=f"[{i}]",
                color=colors[i]
            )
            ax_outputs.scatter(
                len(outputs) - 1,
                outputs[-1],
                color=colors[i],
                s=100,
                edgecolor="black",
                zorder=5
            )

        legend_elements = [
            plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='-',
                       markersize=6, label=f'[{i}]: {val:.2f}')
            for i, val in enumerate(final_outputs)
        ]
        ax_outputs.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize="x-small",
            title="Final Outputs"
        )

        ax_outputs.set_title("Outputs over Iterations")
        ax_outputs.set_xlabel("Iteration")
        ax_outputs.set_ylabel("Output Value")

        # Loss plot
        ax_loss = fig.add_subplot(gs[2, 0])
        ax_loss.plot(self.losses, label=f"Loss over time (lr = {lr})", color="red")
        ax_loss.scatter(len(self.losses) - 1, self.losses[-1], color="red", s=100, edgecolor="black", zorder=5)
        ax_loss.legend(loc="upper right", fontsize="large")
        ax_loss.set_title("Loss over Iterations")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")

        # Weights plot
        ax_weights = fig.add_subplot(gs[1:, 1])

        # Plot all weights first
        for i in range(self.W1.shape[1]):  # rows (output dimension)
            for j in range(self.W1.shape[0]):  # columns (input dimension)
                ax_weights.plot(
                    self.weight_trajectories[j, i, :],
                    label=f"[{j},{i}]"
                )
                ax_weights.scatter(
                    len(self.weight_trajectories[j, i, :]) - 1,
                    self.weight_trajectories[j, i, -1],
                    s=100,
                    edgecolor="black",
                    zorder=5
                )

        # Create legend with exact matrix dimensions
        handles, labels = ax_weights.get_legend_handles_labels()
        ax_weights.legend(
            handles,
            labels,
            loc='lower left',
            fontsize='x-small',
            ncol=self.W1.shape[1],
            labelspacing=0.1,
            columnspacing=0.5,
            handlelength=1,
            handletextpad=0.3
        )

        ax_weights.set_title("Weights over Iterations")
        ax_weights.set_xlabel("Iteration")
        ax_weights.set_ylabel("Weight Value")

        plt.tight_layout()
        # plt.show()
        plt.savefig("nn_viz.png")


if __name__ == "__main__":
    sg = SimpleGradient()
    # sg.gradient_descent()
