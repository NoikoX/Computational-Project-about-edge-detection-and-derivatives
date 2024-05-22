import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../Images/Sukuna.jpg', cv2.IMREAD_GRAYSCALE)

edges_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)

edges_laplacian = cv2.Laplacian(image, cv2.CV_64F)

edges_sobel = cv2.convertScaleAbs(edges_sobel)
edges_laplacian = cv2.convertScaleAbs(edges_laplacian)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(edges_sobel, cmap='gray'), plt.title('Edges (Sobel)')
plt.subplot(133), plt.imshow(edges_laplacian, cmap='gray'), plt.title('Edges (Laplacian)')
plt.show()
