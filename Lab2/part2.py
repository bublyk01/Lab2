import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image_raw = imread(image_path)

print(image_raw.shape)

plt.imshow(image_raw)
plt.show()
