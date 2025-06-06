import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Optional: Display the first few rows
print(df.head())

# Use only numerical features for clustering
features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['CustomerID'])

# Step 2: Visualize data (Optional PCA to reduce to 2D)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.title('Customers Visualized in 2D (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Step 3: Elbow Method to find optimal K
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Step 4: Fit K-Means using optimal K (you can choose based on elbow graph, e.g., k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(features)

# Add labels to DataFrame
df['Cluster'] = labels

# Visualize clusters in PCA 2D space
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(reduced_features[labels == cluster, 0],
                reduced_features[labels == cluster, 1],
                label=f'Cluster {cluster}')
plt.legend()
plt.title('Customer Segments')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Step 5: Evaluate clustering with Silhouette Score
score = silhouette_score(features, labels)
print(f"Silhouette Score: {score:.3f}")
