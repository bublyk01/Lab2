import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image_raw = imread(image_path)

print(f"Original image: {image_raw.shape}")

image_sum = image_raw.sum(axis=2)

image_bw = image_sum / image_sum.max()

print(f"B&W image: {image_bw.shape}")

print(f"Maximum value in black and white image: {image_bw.max()}")

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_raw)
plt.title("Original image")

plt.subplot(1, 2, 2)
plt.imshow(image_bw, cmap='gray')
plt.title("B&W image")
plt.show()
