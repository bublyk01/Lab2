import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image_path = 'image.jpg'
image_raw = imread(image_path)

image_sum = image_raw.sum(axis=2)

image_bw = image_sum / image_sum.max()

image_flattened = image_bw.reshape(-1, image_bw.shape[1])

components_quantity = [5, 15, 50, 100, 200, 400]

plt.figure(figsize=(18, 12))

for i, n_components in enumerate(components_quantity, 1):
    pca = PCA(n_components=n_components)
    image_reduced = pca.fit_transform(image_flattened)

    image_reconstructed = pca.inverse_transform(image_reduced)

    image_reconstructed = image_reconstructed.reshape(image_bw.shape)

    plt.subplot(2, 3, i)
    plt.imshow(image_reconstructed, cmap='gray')
    plt.title(f"{n_components} used")
    plt.axis('off')

plt.tight_layout()
plt.show()
