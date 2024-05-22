import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('../Images/images.png')

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')

vertical_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

n, m, d = img.shape
vertical_edges_img = np.zeros_like(img)
for row in range(1, n - 1):
    for col in range(1, m - 1):
        local_pixels = img[row - 1: row + 2, col - 1: col + 2, 0]
        transformed_pixels = vertical_filter * local_pixels
        vertical_score = (transformed_pixels.sum() + 4) / 8
        vertical_edges_img[row, col] = np.array([vertical_score] * 3 + [1])


plt.subplot(1, 2, 2)
plt.imshow(vertical_edges_img)
plt.title('Vertical Edges Image')

plt.show()
