import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# -----------------------------
# K-Means from Scratch
# -----------------------------
class KMeansScratch:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) for i in range(self.k)
            ])

            # Stop if centroids do not change
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def calculate_wcss(self, X):
        wcss = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return wcss


# -----------------------------
# Data Generation
# -----------------------------
X, true_labels = make_blobs(
    n_samples=500,
    centers=4,
    n_features=4,
    cluster_std=1.2,
    random_state=42
)

# -----------------------------
# Elbow Method
# -----------------------------
wcss_values = []
k_values = range(1, 8)

for k in k_values:
    model = KMeansScratch(k)
    model.fit(X)
    wcss = model.calculate_wcss(X)
    wcss_values.append(wcss)

# Print WCSS values
print("Elbow Method Data (K vs WCSS):")
for k, wcss in zip(k_values, wcss_values):
    print(f"K = {k}, WCSS = {wcss:.2f}")

# -----------------------------
# Final Clustering (Chosen K)
# -----------------------------
optimal_k = 4
final_model = KMeansScratch(optimal_k)
final_model.fit(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot clusters
plt.figure(figsize=(7, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=final_model.labels, cmap='viridis', s=30)
plt.title("K-Means Clustering (K=4)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
