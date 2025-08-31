import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset into DataFrame
## Dataset url "https://www.kaggle.com/datasets/kukuroo3/body-performance-data"
data = pd.read_csv("abalone.csv")

# Step 2: Drop Sex and Rings columns as requested
data_features = data.drop(columns=["Sex", "Rings"])
print(data_features.head())

# Define X as the remaining features (all numerical)
X = data_features

# Step 3: Standardize ALL numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", sum(pca.explained_variance_ratio_))

# Step 5: Visualize PCA result (without color coding since we removed target variables)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Abalone Dataset (Physical Measurements Only)')
plt.grid(True, alpha=0.3)
plt.show()
