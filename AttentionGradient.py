import numpy as np
from sklearn.preprocessing import Normalizer
import pandas as pd


def softmax(Z, axis=0):
    # Subtract max for numerical stability
    Z_shifted = Z - np.max(Z, axis=axis, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)


class AttentionGradient:
    def __init__(self, x_size, num_x):
        self.z2 = None
        self.a1 = None
        self.normNX_2 = None
        self.z1 = None
        self.normNX = None
        self.NX_X = None
        self.X = []
        for _ in range(num_x):
            x = np.random.uniform(-100, 100, size=x_size)
            x = (x - x.mean()) / x.std()
            self.X.append(x)
        self.X = np.array(self.X).T

        self.Q = np.random.randn(x_size // 2, x_size) * 0.01
        self.K = np.random.randn(x_size // 2, x_size) * 0.01
        self.QK = np.zeros((x_size // 2, x_size // 2))
        self.SMQK = np.zeros((x_size // 2, x_size // 2))

        self.V = np.random.randn(x_size // 2, x_size) * 0.01

        self.NX = np.zeros((x_size, num_x))

        self.FFN_W1 = np.random.randn(x_size, x_size) * 0.01

        self.FFN_W2 = np.random.randn(x_size, x_size) * 0.01

        self.EX = np.zeros((x_size, num_x))
        self.ran = False

        self.TargetMat = self.generate_target_matrix()

        self.Loss = None

        #################
        # Gradient

        self.dQ = self.dK = self.dV = self.dFFN_W1 = self.dFFN_W2 = None

    def generate_target_matrix(self):
        """
        Generate a probabilistic TargetMat based on X.
        """
        # Use a simple linear transformation of X
        W_target = np.random.randn(self.X.shape[0], self.X.shape[0]) * 0.01
        target_raw = W_target @ self.X  # Shape: (x_size, num_x)

        # Apply softmax across columns to ensure a probabilistic interpretation
        target_normalized = softmax(target_raw, axis=0)

        return target_normalized

    def attention(self, verbose=False):
        def printLog(x):
            if verbose:
                print(x)

        q, k = self.Q @ self.X, self.K @ self.X
        printLog(f"\n---MatMul Q*X({q.shape}) and K*X({k.shape})")
        self.QK = q @ k
        # need to add masking here !

        printLog(f"\n---pre-SoftMax QK ({self.QK.shape})")
        self.SMQK = softmax(self.QK.T).T
        printLog(f"\tbecomes SMQK({self.SMQK.shape})")

        printLog(f"\n---MatMul SMQK({self.SMQK.shape}) with V({self.V.shape}) and then Transpose")
        self.NX = (self.SMQK @ self.V).T
        printLog(f"\tbecomes NX({self.NX.shape})")
        printLog(f"\n---Now we add NX({self.NX.shape}) and X({self.X.shape})")
        self.NX_X = self.NX + self.X

        printLog(f"\n---pre-normalized NX_X({self.NX_X.shape})")
        self.normNX = Normalizer().fit_transform(self.NX_X)
        printLog(f"\t becomes NX({self.normNX.shape})")

        printLog(f"\n---MatMul FFN_W1({self.FFN_W1.shape}) and normNX({self.normNX.shape})")
        self.z1 = self.FFN_W1 @ self.normNX

        printLog(f"\n---pre-SoftMax z1({self.z1.shape})")
        self.a1 = softmax(self.z1.T).T
        printLog(f"\tbecomes a1({self.a1.shape})")

        printLog(f"\n---MatMul FFN_W2({self.FFN_W2.shape}) and a1({self.a1.shape})")
        self.z2 = self.FFN_W2 @ self.a1

        printLog(f"\n---adding normNX({self.normNX.shape}) and z2({self.z2.shape})")
        self.normNX_2 = self.normNX + self.z2

        printLog(f"\n---pre-normalized EX({self.normNX_2.shape})")
        self.EX = Normalizer().fit_transform(self.normNX_2)
        self.ran = True
        printLog(f"\tbecomes EX({self.EX.shape})\n\n\nATTENTION DONE\n\n\n\n\n\n")

        printLog(f"Input X:\n\n{pd.DataFrame(self.X)}")
        printLog(
            f"\nBecomes Enhanced X:\n\n{pd.DataFrame(self.EX)}\n\n"
            f"****************************************************************************************")

    def calculate_loss(self):
        if not self.ran:
            self.attention()

        # Normalize TargetMat for a proper probabilistic interpretation
        norm_target_mat = softmax(self.TargetMat, axis=0)

        # Add small epsilon to EX to avoid log(0)
        eps = 1e-12  # Small constant for numerical stability
        safe_EX = np.clip(self.EX, eps, None)  # Ensure EX >= eps

        # Compute cross-entropy loss
        self.Loss = -np.sum(norm_target_mat * np.log(safe_EX)) / self.EX.size
        
    def backprop(self):
        if self.Loss is None:
            self.calculate_loss()
        


if __name__ == "__main__":
    ag = AttentionGradient(10, 5)

    ag.attention(verbose=True)  # Enable detailed logging
    ag.calculate_loss()
    print(f"Target Matrix: \n{ag.TargetMat}")
    print(f"\nCross-Entropy Loss: {ag.Loss:.6f}")

