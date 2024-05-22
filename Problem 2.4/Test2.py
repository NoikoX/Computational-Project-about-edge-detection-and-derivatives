import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('../Images/asuka.jpg', cv2.IMREAD_GRAYSCALE)

# here i compute the x and y derivatives
dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# here is just linear combination of derivative
edges = np.hypot(dx, dy)
edges = edges / edges.max() * 255

magnitude = cv2.magnitude(dx, dy)
direction = cv2.phase(dx, dy)


quantized_direction = np.round(direction / (np.pi / 4)) % 4


histogram, bins = np.histogram(quantized_direction, bins=4, range=(0, 4))


histogram = histogram / histogram.sum()
plt.figure(figsize=(10, 10))

plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.bar(bins[:-1], histogram, width=0.8)
plt.title('Histogram of Gradient Directions'), plt.xticks(bins[:-1], labels=['0', '45', '90', '135'])

plt.show()
