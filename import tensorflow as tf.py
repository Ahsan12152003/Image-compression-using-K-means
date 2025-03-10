import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans

# Load dataset (using MNIST as an example)
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(len(x_train), -1)  # Flatten images
x_test = x_test.reshape(len(x_test), -1)  # Flatten images

# Reduce dataset size for faster computation
x_train = x_train[:2000]  # Reduced further
x_test = x_test[:500]  # Reduced further

# Apply K-Means clustering
num_clusters = 16  # Reduced clusters for faster processing
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5)
kmeans.fit(x_train)

# Compress images by replacing each pixel value with its cluster centroid
compressed_train = kmeans.predict(x_train)
compressed_test = kmeans.predict(x_test)

# Reconstruct images from cluster centers
x_train_compressed = kmeans.cluster_centers_[compressed_train].reshape(-1, 28, 28)
x_test_compressed = kmeans.cluster_centers_[compressed_test].reshape(-1, 28, 28)

# Display original and compressed images
n = 5  # Display fewer images for faster visualization
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_compressed[i], cmap='gray')
    plt.axis('off')
plt.show()
