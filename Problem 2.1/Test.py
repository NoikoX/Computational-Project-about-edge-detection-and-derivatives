import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../Images/Butterfly.jpg', cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_image, 100, 200)

overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

overlay[edges != 0] = [230,130, 0]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Edges Detected')
plt.axis('off')

plt.show()
