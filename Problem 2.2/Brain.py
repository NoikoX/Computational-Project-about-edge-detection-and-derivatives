import cv2
import numpy as np

image = cv2.imread('../Images/MRI_brain.jpg', cv2.IMREAD_GRAYSCALE)

# heree i use canny edge detection
edges = cv2.Canny(image, 30, 150)

stacked_image = np.hstack((image, edges))

cv2.imshow('Original vs Edge Detected', stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
