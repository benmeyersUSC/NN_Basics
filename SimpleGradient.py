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


class SimpleGradient:
    def __init__(self, x_size=16, y_size=10):
        self.X = np.random.uniform(-100, 100, size=x_size)
        self.X = (self.X - self.X.mean()) / self.X.std()

        # self.W1 = np.random.randn(y_size, self.X.size) * 10
        self.W1 = np.zeros((y_size, self.X.size))

        self.Y = self.W1 @ self.X

        self.target = np.random.uniform(-100, 100, size=self.Y.size)

        self.loss = sum((self.Y - self.target) ** 2)

        print(
            f"\n\nINITIALIZATION:"
            f"\n************************************************************************************\n"
            f"{blue_print("X")}: {pd.DataFrame(self.X).T}, "
            f"\n{orange_print("W1")}: {pd.DataFrame(self.W1)}, "
            f"\n{yellow_print("Output")}: {pd.DataFrame(self.Y).T}, "
            f"\n{green_print("Target")}: {pd.DataFrame(self.target).T}, "
            f"\n{red_print("Loss")}: {self.loss}"
        f"\n************************************************************************************")

        self.losses = self.weight_trajectories = self.output_trajectories = None

    def gradient_descent(self, iterations=100, learning_rate=0.001, plot=True):
        self.losses = [self.loss]
        self.weight_trajectories = np.zeros((self.W1.shape[0], self.W1.shape[1], iterations))
        self.output_trajectories = np.zeros((self.Y.size, iterations))

        self.weight_trajectories[:, :, 0] = self.W1
        self.output_trajectories[:, 0] = self.Y

        for iteration in range(1, iterations):
            # Gradient computation
            dloss_dY = 2 * (self.Y - self.target)  # Gradient wrt outputs
            dloss_dW1 = np.outer(dloss_dY, self.X)  # Gradient wrt weights
            if iteration % 25 == 0:
                print(
                    f"\n\n------------------------------------------------------------------------------------------------")
                print(f"Iteration {iteration}:")
                print(
                    f"{blue_print("X")}: {pd.DataFrame(self.X).T}, \n{yellow_print("Output")}: {pd.DataFrame(self.Y).T}, \n{green_print("Target")}: {pd.DataFrame(self.target).T}, \n{red_print("Loss")}: {self.loss}")
                print(
                    f"\n\t{cyan_print("Gradient wrt Outputs")}: (d-{red_print("Loss")} / d-{yellow_print("Output")}): \n\n\t\t{pd.DataFrame(dloss_dY).T}\n\n\tFormula: (2 * ({yellow_print("Output")} - {green_print("Target")}))")
                print(
                    f"\n\t{magenta_print("Gradient wrt Weights")}: (d-({cyan_print("Gradient wrt Outputs")}) / d-{blue_print("X")}): \n{pd.DataFrame(dloss_dW1)}\n\n\tFormula: (OuterProduct ({cyan_print("Gradient wrt Outputs")}) x {blue_print("X")})")
                print(
                    f"------------------------------------------------------------------------------------------------")

            # Update weights
            self.W1 -= learning_rate * dloss_dW1
            self.weight_trajectories[:, :, iteration] = self.W1  # Store updated weights

            # Recompute predictions and loss
            self.Y = self.W1 @ self.X
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
    sg.gradient_descent()
