import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('../Images/asuka.jpg', cv2.IMREAD_GRAYSCALE)


sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
edges = np.hypot(sobelx, sobely)

harris_corners = cv2.cornerHarris(image, 2, 3, 0.04)

plt.figure(figsize=(10, 10))

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(harris_corners, cmap='gray')
plt.title('Feature Image'), plt.xticks([]), plt.yticks([])

plt.show()
