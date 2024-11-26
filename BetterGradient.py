import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fake_data as fkd
green_print = lambda x: f"\033[30;42m{x}\033[0m"
red_print = lambda x: f"\033[30;41m{x}\033[0m"
orange_print = lambda x: f"\033[1;33;40m{x}\033[0m"
yellow_print = lambda x: f"\033[30;43m{x}\033[0m"
blue_print = lambda x: f"\033[30;44m{x}\033[0m"
magenta_print = lambda x: f"\033[30;45m{x}\033[0m"
cyan_print = lambda x: f"\033[30;46m{x}\033[0m"



def softmax(Z):
    # Subtract max for numerical stability (prevents overflow)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def ReLU(Z):
    return np.maximum(0, Z)


def deriv_ReLU(Z):
    return Z > 0  # Returns a boolean array, which is implicitly converted to 0s and 1s


class BetterGradient:
    def __init__(self, x_size=16, y_size=10):

        self.X = np.random.uniform(-100, 100, size=x_size)
        self.X = (self.X - self.X.mean()) / self.X.std()

        self.W1 = np.random.randn(y_size, x_size) * 0.01  # Random small weights
        self.A1 = np.zeros(y_size)

        self.W2 = np.random.randn(y_size, y_size) * 0.01  # Square matrix
        self.Z2 = None

        self.Y = np.zeros(y_size)

        self.target = np.zeros(self.Y.size)
        self.target[random.randint(0, len(self.target) - 1)] = 1

        self.loss = None
        self.forward_pass()
        self.calculate_loss()

        self.losses = self.weight1_trajectories = self.weight2_trajectories = self.a1_trajectories = self.output_trajectories = None


        print(
            f"\n\nINITIALIZATION:"
            f"\n************************************************************************************\n"
            f"{blue_print("X")}: {pd.DataFrame(self.X).T}, "
            f"\n{orange_print("W1")}: {pd.DataFrame(self.W1)}, "
            f"\n{magenta_print("W2")}: {pd.DataFrame(self.W2)}, "
            f"\n\n{green_print("Target")}: {pd.DataFrame(self.target)}, "
            f"\n************************************************************************************")

        self.losses = self.weight1_trajectories = self.weight2_trajectories = self.output_trajectories = None

    def forward_pass(self):
        x = self.X
        self.A1 = np.maximum(self.W1 @ x, 0)  # ReLU
        self.Z2 = self.W2 @ self.A1
        self.Y = softmax(self.Z2.reshape(1, -1))

    def calculate_loss(self):
        self.loss = -np.sum(self.target * np.log(self.Y + 1e-12))  # Add epsilon for numerical stability

    def gradient_descent(self, iterations=500, learning_rate=0.001, plot=True, insert_data="", make_data=False):
        in_data = inX = inY = None
        data_good = False

        if len(insert_data) > 4:
            if make_data:
                # Generate dataset
                X, y = fkd.generate_complex_dataset(n_samples=5427, noise_level=0.3)

                fkd.export_dataset_to_csv(
                    # X, y,
                    *fkd.generate_complex_dataset(n_samples=5427, noise_level=0.3),
                    filename=insert_data,
                    feature_prefix='x_',
                    target_column='label'
                )

            if ".csv" in insert_data:
                try:
                    in_data = pd.read_csv(insert_data)
                    in_data = in_data.drop_duplicates()
                    in_data = in_data.dropna()
                    inX = in_data.drop(columns=["label"])
                    inY = in_data["label"]
                    inY_one_hot = np.zeros((len(inY), 10))
                    inY_one_hot[np.arange(len(inY)), inY.values] = 1
                    inY = inY_one_hot
                    data_good = True
                except:
                    pass
        else:
            if make_data:
                # Generate dataset
                # X, y = generate_complex_dataset(n_samples=5427, noise_level=0.3)

                fkd.export_dataset_to_csv(
                    # X, y,
                    # X, y,
                    *fkd.generate_complex_dataset(n_samples=5427, noise_level=0.3),
                    filename="NN_data.csv",
                    feature_prefix='x_',
                    target_column='label'
                )



        self.losses = [self.loss]
        self.weight1_trajectories = np.zeros((self.W1.shape[0], self.W1.shape[1], iterations))
        self.weight2_trajectories = np.zeros((self.W2.shape[0], self.W2.shape[1], iterations))
        self.a1_trajectories = np.zeros((self.A1.shape[0], iterations))
        self.output_trajectories = np.zeros((self.Y.size, iterations))

        self.weight1_trajectories[:, :, 0] = self.W1
        self.weight2_trajectories[:, :, 0] = self.W2
        self.a1_trajectories[:, 0] = self.A1
        self.output_trajectories[:, 0] = self.Y

        for i in range(1, iterations):
            if data_good:
                rNum = np.random.randint(len(in_data))
                self.X = inX.iloc[rNum].values
                self.Y = inY[rNum]


            self.forward_pass()
            self.calculate_loss()

            # Gradients for cross-entropy loss
            dL_dY = (self.Y - self.target).reshape(-1)
            """
            see SoftMaxDerivative.py
            """

            # Backpropagation through W2
            dL_dW2 = np.outer(dL_dY, self.A1)
            """
            dZ2/dW2 = A1 (input to this layer)
            
            so this is dL_dW2 = dL_dY * dZ2_dW2 = dL_dY * A1
            """

            # Backpropagation through ReLU and W1
            dZ2_dA1 = self.W2.T  # Gradient from second layer weights
            """
            Z2 = W2 @ A1, hence dZ2/dA1 = W2 (transpose is to fit with necessary matmul)
            """

            dA1_dZ1 = (self.A1 > 0).astype(float)  # ReLU derivative
            """
            A1 = ReLU(Z1) and ReLU(x) = max(0, x), hence dA1/dZ1 = ReLU'(Z1), which is a boolean vector of scalars (1)
            """

            dL_dA1 = dZ2_dA1 @ dL_dY
            """
            dA1 affects dZ2/dA1 (W2), which then goes through loss gradient, dL/dY
            """

            dL_dZ1 = dL_dA1 * dA1_dZ1
            """
            * because same dimension (now using the boolean vector to turn on/off multiplications)
            dL/dZ1 is just dL/dA1 when Z1 is positive
            """

            dL_dW1 = np.outer(dL_dZ1, self.X)  # dZ1/dW1 = X (input to first layer)
            """
            dL/dZ1 encompasses all that comes after Z1, so we look to ahead of W1
            Z1 = W1 @ X, hence dL/dW1 includes X as a term which affects dL/dZ1 
            """

            self.W1 -= learning_rate * dL_dW1
            self.W2 -= learning_rate * dL_dW2

            self.weight1_trajectories[:, :, i] = self.W1
            self.weight2_trajectories[:, :, i] = self.W2
            self.a1_trajectories[:, i] = self.A1

            self.output_trajectories[:, i] = self.Y
            self.losses.append(self.loss)


        if plot:
            self._plot_gd(learning_rate)


    def _plot_gd(self, lr):
        plt.style.use('ggplot')  # A built-in matplotlib style
        fig = plt.figure(figsize=(20, 15), facecolor='#f0f0f0')
        gs = GridSpec(3, 2, height_ratios=[2, 2, 1])

        # Color palette
        colors = plt.cm.cool(np.linspace(0, 1, max(self.W1.shape[1], self.W2.shape[1])))

        # W1 Weights Trajectory - Large Subplot
        ax_weights1 = fig.add_subplot(gs[0, 0])
        ax_weights1.set_title('W1 Weights Trajectory', fontsize=16, fontweight='bold')
        for i in range(self.W1.shape[1]):
            for j in range(self.W1.shape[0]):
                ax_weights1.plot(
                    self.weight1_trajectories[j, i, :],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=f'W1[{j},{i}]'
                )
        ax_weights1.set_xlabel('Iteration', fontsize=12)
        ax_weights1.set_ylabel('Weight Value', fontsize=12)
        ax_weights1.grid(True, linestyle='--', alpha=0.5)

        # W2 Weights Trajectory - Large Subplot
        ax_weights2 = fig.add_subplot(gs[0, 1])
        ax_weights2.set_title('W2 Weights Trajectory', fontsize=16, fontweight='bold')
        for i in range(self.W2.shape[1]):
            for j in range(self.W2.shape[0]):
                ax_weights2.plot(
                    self.weight2_trajectories[j, i, :],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                    label=f'W2[{j},{i}]'
                )
        ax_weights2.set_xlabel('Iteration', fontsize=12)
        ax_weights2.set_ylabel('Weight Value', fontsize=12)
        ax_weights2.grid(True, linestyle='--', alpha=0.5)

        # Output Trajectory
        ax_outputs = fig.add_subplot(gs[1, 0])
        ax_outputs.set_title('Outputs over Iterations', fontsize=16, fontweight='bold')
        for i in range(self.Y.size):
            outputs = self.output_trajectories[i, :]
            ax_outputs.plot(
                outputs,
                color=plt.cm.viridis(i / self.Y.size),
                linewidth=2,
                label=f'Output [{i}]'
            )
        ax_outputs.set_xlabel('Iteration', fontsize=12)
        ax_outputs.set_ylabel('Output Value', fontsize=12)
        ax_outputs.grid(True, linestyle='--', alpha=0.5)
        ax_outputs.legend(loc='best', fontsize=10)

        # Loss Trajectory
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_loss.set_title(f'Loss over Iterations (lr = {lr})', fontsize=16, fontweight='bold')
        ax_loss.plot(
            self.losses,
            color='crimson',
            linewidth=3,
            label='Loss'
        )
        ax_loss.set_xlabel('Iteration', fontsize=12)
        ax_loss.set_ylabel('Loss Value', fontsize=12)
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        # Target Distribution
        ax_target = fig.add_subplot(gs[2, :])
        ax_target.set_title('Target Distribution', fontsize=16, fontweight='bold')
        ax_target.imshow(
            np.expand_dims(self.target, axis=0),
            cmap='coolwarm',
            aspect='auto',
            interpolation='nearest'
        )
        ax_target.set_xlabel('Output Index', fontsize=12)
        ax_target.set_xticks(range(len(self.target)))
        ax_target.set_xticklabels([f'[{i}]' for i in range(len(self.target))])
        ax_target.set_yticks([])

        # Final layout adjustments
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.suptitle('Gradient Descent Visualization', fontsize=20, fontweight='bold', y=1.02)
        plt.savefig("BetterGradientPlot.png", dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    bg = BetterGradient()
    bg.gradient_descent(iterations=2927, insert_data="NN_data.csv")


