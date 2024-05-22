import cv2
import numpy as np


image = cv2.imread('../Images/asuka.jpg', cv2.IMREAD_GRAYSCALE)

# so here i define sobel matrices for horizontal and vertical edges
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# and then i do the experiment with different combinations of coefficients
coefficients = [0.5, 0.5]

# here i apply linear combination of derivatives
edges_x = cv2.filter2D(image, -1, sobel_x)
edges_y = cv2.filter2D(image, -1, sobel_y)
edges_combined = coefficients[0] * edges_x + coefficients[1] * edges_y


cv2.imshow('Original Image', image)
cv2.imshow('Combined Edges', edges_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
