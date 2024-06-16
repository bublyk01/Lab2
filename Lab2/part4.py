import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image_path = 'image.jpg'
image_raw = imread(image_path)

image_sum = image_raw.sum(axis=2)

image_bw = image_sum / image_sum.max()

image_flattened = image_bw.reshape(-1, image_bw.shape[1])

pca = PCA()
pca.fit(image_flattened)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for a 95% value: {n_components}")

pca = PCA(n_components=n_components)
image_reduced = pca.fit_transform(image_flattened)

image_reconstructed = pca.inverse_transform(image_reduced)

image_reconstructed = image_reconstructed.reshape(image_bw.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_bw, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(image_reconstructed, cmap='gray')

plt.show()
