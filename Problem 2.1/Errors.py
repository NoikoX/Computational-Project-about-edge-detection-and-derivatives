import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('../Images/asuka.jpg', cv2.IMREAD_GRAYSCALE)

# soo here I define finite difference kernels with different levels of error
# the first onee is finite difference with low truncation error O(h^2)
kernel_low_error = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

# and the second one is finite difference with high truncation error O(h)
kernel_high_error = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])


edges_low_error = cv2.filter2D(image, -1, kernel_low_error)
edges_high_error = cv2.filter2D(image, -1, kernel_high_error)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(edges_low_error, cmap='gray'), plt.title('Edges (Low Error)')
plt.subplot(133), plt.imshow(edges_high_error, cmap='gray'), plt.title('Edges (High Error)')
plt.show()
