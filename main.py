import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print(df.head())

features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['CustomerID'])

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.title('Customers Visualized in 2D (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

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

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(features)

df['Cluster'] = labels

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

score = silhouette_score(features, labels)
print(f"Silhouette Score: {score:.3f}")
