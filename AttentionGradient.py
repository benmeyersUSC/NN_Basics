
import numpy as np
from sklearn.preprocessing import Normalizer
import pandas as pd


def softmax(Z, axis=0):
    """Softmax function, applied along a specified axis."""
    Z_shifted = Z - np.max(Z, axis=axis, keepdims=True)  # Subtract max for numerical stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)


def ReLU(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)


class AttentionGradient:
    def __init__(self, x_size, num_x):
        self.Loss = None
        self.X = self.initialize_inputs(x_size, num_x)  # Tokens as columns
        self.Q = np.random.rand(x_size, x_size)
        self.K = np.random.rand(x_size, x_size)
        self.V = np.random.rand(num_x, x_size)
        self.FFN_W1 = np.random.rand(x_size, x_size)
        self.FFN_W2 = np.random.rand(x_size, x_size)
        self.TargetMat = self.generate_target_matrix()  # Target probabilities for loss
        self.EX = None  # Final enhanced X
        self.ran = False  # Whether attention has been run yet

    @staticmethod
    def initialize_inputs(x_size, num_x):
        """Initialize input matrix with normalized random data."""
        X = []
        for _ in range(num_x):
            x = np.random.uniform(-99999, 99999, size=x_size)
            x = (x - x.mean()) / x.std()  # Normalize each input vector
            X.append(x)
        X = np.array(X).T
        print(f"Initialized X (tokens as columns):\n{pd.DataFrame(X)}")
        return X

    def generate_target_matrix(self):
        """Generate probabilistic TargetMat based on X."""
        W_target = np.random.randn(self.X.shape[0], self.X.shape[0]) * 0.01
        target_raw = W_target @ self.X
        return softmax(target_raw, axis=0)  # Column-wise softmax

    @staticmethod
    def mask(matrix: np.ndarray):
        # Create a lower-triangular mask (1s below the diagonal, 0s elsewhere)
        mask = np.triu(np.ones(matrix.shape), k=0)
        # Apply the mask by replacing 0s with -inf
        return np.where(mask == 0, float('-inf'), matrix)


    def attention(self, verbose=False):
        """Run the forward pass of the attention block."""
        def printLog(msg):
            if verbose:
                print(msg)

        # Step 1: Compute QK and apply softmax
        q, k = self.Q @ self.X, self.K @ self.X  # Q*X and K*X
        printLog(f"{"*"*54}\nStep 1: Compute Q.K and apply SoftMax")
        printLog(f"q=Q @ X: \n{pd.DataFrame(q)}")
        printLog(f"\nk = K @ X:\n{pd.DataFrame(k)}")
        QK = q.T @ k  # (num_x x num_x)
        printLog(f"\nQK = q.T @ k:\n{pd.DataFrame(QK)}")
        QK = self.mask(QK)
        printLog(f"\nMASKED***QK = MASK[QK]:\n{pd.DataFrame(QK)}")
        SMQK = softmax(QK.T).T  # Column-wise softmax over attention scores
        printLog(f"Attention Pattern:\n\n{pd.DataFrame(SMQK)}")
        printLog("*"*54)

        # Step 2: Compute attention output and add residual connection
        printLog(f"{"*" * 54}\nStep 2: Compute Attention output and add residual (X)")
        NX = (SMQK @ self.V).T  # Shape: (x_size x num_x)
        printLog(f"\nNX = [SMQK @ V].T:\n{pd.DataFrame(NX)}")
        NX_X = NX + self.X  # Residual connection
        printLog(f"\nNX_X = NX + X:\n{pd.DataFrame(NX_X)}")
        printLog("*" * 54)

        # Step 3: Normalize (column-wise) NX_X
        printLog(f"{"*" * 54}\nStep 3: Normalize NX_X column/tokenwise")
        normNX = Normalizer().fit_transform(NX_X.T).T  # Token-wise normalization
        printLog(f"\nnormNX = Norm(NX_X.T).T:\n{pd.DataFrame(normNX)}")
        printLog("*" * 54)

        # Step 4: Feed-forward network (FFN)
        printLog(f"{"*" * 54}\nStep 4: Feed-Forward Neural Network (FFN)")
        z1 = self.FFN_W1 @ normNX  # First layer: Linear transformation
        printLog(f"\nz1 = FFN_W1 @ normNX:\n{pd.DataFrame(z1)}")
        a1 = ReLU(z1)  # Apply ReLU column-wise
        printLog(f"\na1 = ReLU(z1):\n{pd.DataFrame(a1)}")


        z2 = self.FFN_W2 @ a1  # Second layer: Linear transformation
        printLog(f"\nz2 = FFN_W2 @ a1:\n{pd.DataFrame(z2)}")
        normNX_2 = normNX + z2  # Add residual connection
        printLog(f"\nnormNX_2 = normNX + z2:\n{pd.DataFrame(normNX_2)}")
        printLog("*" * 54)

        # Step 5: Final softmax (column-wise)
        printLog(f"{"*" * 54}\nStep 5: Final SoftMax column/tokenwise")
        self.EX = softmax(normNX_2, axis=0)  # Token-wise probabilities
        self.ran = True
        printLog(f"\nEX = SoftMax(normNX_2):\n{pd.DataFrame(self.EX)}")
        printLog("*" * 54)

    def calculate_loss(self):
        """Calculate cross-entropy loss."""
        if not self.ran:
            self.attention()

        eps = 1e-12  # Small constant for numerical stability
        safe_EX = np.clip(self.EX, eps, None)  # Avoid log(0)
        self.Loss = -np.sum(self.TargetMat * np.log(safe_EX)) / self.EX.size
        print(f"Cross-Entropy Loss: {self.Loss:.6f}")


if __name__ == "__main__":
    # Parameters
    x_size = 10  # Size of each input vector
    num_x = 5   # Number of tokens (columns)

    # Instantiate and run
    ag = AttentionGradient(x_size, num_x)
    ag.attention(verbose=True)  # Forward pass with detailed logs
    ag.calculate_loss()         # Calculate loss

    # Display results
    print(f"Target Matrix: \n{pd.DataFrame(ag.TargetMat)}")
    print(f"Enhanced X (EX): \n{pd.DataFrame(ag.EX)}")

    print(f"Loss: {ag.Loss}")
