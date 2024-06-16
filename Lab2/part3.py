import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image_path = 'image.jpg'
image_raw = imread(image_path)

print(f"Original image shape: {image_raw.shape}")

image_sum = image_raw.sum(axis=2)

image_bw = image_sum / image_sum.max()

print(f"Black and white image shape: {image_bw.shape}")

max_value_bw = image_bw.max()
print(f"Maximum value in black and white image: {max_value_bw}")

image_flattened = image_bw.reshape(-1, image_bw.shape[1])

pca = PCA()
pca.fit(image_flattened)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

n_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Number of components needed for a 95% value: {n_components}")

plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axvline(x=n_components, color='k', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
