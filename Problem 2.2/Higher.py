import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../Images/gojo.jpg', cv2.IMREAD_GRAYSCALE)

image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

laplacian = cv2.Laplacian(image_blurred, cv2.CV_64F)

laplacian = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian of Gaussian')
plt.show()
