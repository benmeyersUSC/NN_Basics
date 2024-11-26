import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_complex_dataset(n_samples=1000, noise_level=0.2, rseed=0):
    """
    Generate a synthetic dataset with 16 features and 10 categories
    using complex non-linear transformations and added noise.

    Parameters:
    - n_samples: Number of data points
    - noise_level: Standard deviation of Gaussian noise

    Returns:
    - X: Feature matrix
    - y: Categorical labels
    """
    if rseed == 0:
        rseed = np.random.randint(9999999)
    # Set random seed for reproducibility
    np.random.seed(rseed)


    # Generate base features with complex interactions
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.sin(x1) * 2
    x3 = np.exp(x1 / 5) - 1
    x4 = np.log(x1 + 1)
    x5 = x1 ** 2 / 10
    x6 = np.cos(x1) * 3
    x7 = np.sqrt(np.abs(x1))
    x8 = np.arctan(x1)

    # More complex feature interactions
    x9 = x1 * x2 + np.random.normal(0, noise_level, n_samples)
    x10 = x3 ** 2 - x4 + np.random.normal(0, noise_level, n_samples)
    x11 = np.sin(x5) * np.cos(x6) + np.random.normal(0, noise_level, n_samples)
    x12 = x7 * np.exp(x8 / 3) + np.random.normal(0, noise_level, n_samples)
    x13 = np.tanh(x1 / 2) * np.log(x2 + 2 + 0.01) + np.random.normal(0, noise_level, n_samples)
    x14 = np.power(x3, 1 / 3) - np.sqrt(x4) + np.random.normal(0, noise_level, n_samples)
    x15 = np.sin(x5) * x6 + np.random.normal(0, noise_level, n_samples)
    x16 = np.arcsinh(x1 / 3) + np.random.normal(0, noise_level, n_samples)

    # Create feature matrix
    X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8,
                         x9, x10, x11, x12, x13, x14, x15, x16])

    # Create categorical labels based on a complex decision boundary
    y = np.zeros(n_samples, dtype=int)

    # 10 categories with different decision boundaries
    y[(x1 > 5) & (x2 > 0)] = 1
    y[(x3 < 2) & (x4 > 1)] = 2
    y[(x5 > 10) & (x6 < -1)] = 3
    y[(np.sin(x7) > 0.5) & (x8 > 2)] = 4
    y[(x9 < 0) & (x10 > 5)] = 5
    y[(x11 > 1) & (x12 < -2)] = 6
    y[(x13 < -0.5) & (x14 > 3)] = 7
    y[(x15 < 0) & (x16 > 2)] = 8
    y[x1 ** 2 + x2 ** 2 > 50] = 9

    # Normalize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    return X, y


def export_dataset_to_csv(X, y, filename='complex_synthetic_dataset.csv',
                          feature_prefix='feature_',
                          target_column='category'):
    """
    Export the generated dataset to a CSV file.

    Parameters:
    - X: Feature matrix
    - y: Categorical labels
    - filename: Output CSV filename
    - feature_prefix: Prefix for feature column names
    - target_column: Name of the target/category column

    Returns:
    - DataFrame of the exported dataset
    """
    # Create column names
    columns = [f'{feature_prefix}{i + 1}' for i in range(X.shape[1])]

    # Create pandas DataFrame
    df = pd.DataFrame(X, columns=columns)
    df[target_column] = y

    # Save to CSV
    df.to_csv(filename, index=False)

    # Print export details
    print(f"Dataset exported to {filename}")
    print("Dataset Shape:", df.shape)
    print("\nCategory Distribution:")
    print(df[target_column].value_counts())

    return df

if __name__ == "__main__":
    # Generate dataset
    X, y = generate_complex_dataset(n_samples=5427, noise_level=0.54)

    # Optional: Export with custom settings
    custom_df = export_dataset_to_csv(
        X, y,
        filename='NN_data.csv',
        feature_prefix='x_',
        target_column='label'
    )