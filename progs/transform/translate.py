import cv2 
import numpy as np 
import matplotlib.pyplot as plt

image = cv2.imread('ghanari.jpg') 


height, width = image.shape[:2] 
dim = (height, width)

trans_x, trans_y = width / 4, height / 4
T = np.float32([[1, 0, trans_x], 
                [0, 1, trans_y]]) 
img_translation = cv2.warpAffine(image, T, (width, height)) 


img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_translation_rgb = cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB)
# Display the images using matplotlib
plt.figure(figsize=(15, 10))

# Rotated Image
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title(f"Original Image {dim}")
plt.axis('off')

# Nearest Neighbor Scaling
plt.subplot(1, 2, 2)
plt.imshow(img_translation_rgb)
plt.title(f"Translated by {T} ")
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()

