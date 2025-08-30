import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
data = load_iris()
X = data.data[:, 2:4]  # Selecting petal length and petal width only
true_labels = data.target

# Standardize the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implement K-Means from scratch
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape

    # Initialize centroids randomly from data points
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iters):
        # Assign clusters based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # shape (n_samples, k)
        cluster_labels = np.argmin(distances, axis=1)

        # Calculate new centroids as mean of points in cluster
        new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return cluster_labels, centroids

k = 3
labels, centroids = kmeans(X_scaled, k)

# Plot clusters with centroids
plt.figure(figsize=(8,6))
for i in range(k):
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i+1}', alpha=0.6)

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.title('K-Means Clustering on Iris (Petal Length & Width)')
plt.legend()
plt.show()