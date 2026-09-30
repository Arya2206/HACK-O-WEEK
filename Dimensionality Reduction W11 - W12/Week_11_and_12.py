import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", X.shape)


# --------------------------------------------------
# 2. STANDARDIZE DATA
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully.")


# --------------------------------------------------
# 3. PCA
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:", X_pca.shape)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Explained:",
    pca.explained_variance_ratio_.sum()
)


# --------------------------------------------------
# 4. t-SNE
# --------------------------------------------------

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000
)

X_tsne = tsne.fit_transform(X_scaled)

print("\nt-SNE Shape:", X_tsne.shape)


# --------------------------------------------------
# 5. PCA VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i in range(3):

    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=target_names[i],
        s=50
    )

plt.title("PCA - 2D Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "static/pca_plot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 6. t-SNE VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i in range(3):

    plt.scatter(
        X_tsne[y == i, 0],
        X_tsne[y == i, 1],
        label=target_names[i],
        s=50
    )

plt.title("t-SNE - 2D Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "static/tsne_plot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nPlots generated successfully!")
