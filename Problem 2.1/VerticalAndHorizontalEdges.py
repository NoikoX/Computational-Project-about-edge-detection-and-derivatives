import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('../Images/images.png')

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')

vertical_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
horizontal_filter = vertical_filter.transpose()

n, m, d = img.shape
edges_img = np.zeros_like(img)
for row in range(1, n - 1):
    for col in range(1, m - 1):
        local_pixels = img[row - 1: row + 2, col - 1: col + 2, 0]

        vertical_transformed_pixels = vertical_filter * local_pixels
        vertical_score = (vertical_transformed_pixels.sum() + 4) / 8

        horizontal_transformed_pixels = horizontal_filter * local_pixels
        horizontal_score = (horizontal_transformed_pixels.sum() + 4) / 8

        edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** 0.5
        edges_img[row, col] = [edge_score] * 3 + [1]

edges_img = edges_img / edges_img.max()

plt.subplot(1, 2, 2)
plt.imshow(edges_img)
plt.title('Combined Edges Image')

plt.show()
