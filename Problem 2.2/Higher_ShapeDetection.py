import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('../Images/vector-abstract-circular-map-pattern-with-wavy-lines-black-and-white-topographic-line-contours-simple-round-map-design-TA5HNE.jpg', cv2.IMREAD_COLOR)

image_with_circles = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=10, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
plt.title('Image with Detected Circles')
plt.axis('off')

plt.show()
