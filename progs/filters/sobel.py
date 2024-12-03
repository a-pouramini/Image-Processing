import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('../images/sanjab.jpg', cv2.IMREAD_GRAYSCALE)

# First derivative (Sobel)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradients
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradients
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalize Sobel results for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# Second derivative (Laplacian)
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)  # Compute Laplacian
laplacian = cv2.convertScaleAbs(laplacian)  # Normalize for display

# Visualization
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(sobel_x, cmap='gray')
ax[1].set_title("Sobel - Horizontal Edges")
ax[1].axis('off')

ax[2].imshow(sobel_y, cmap='gray')
ax[2].set_title("Sobel - Vertical Edges")
ax[2].axis('off')

ax[3].imshow(sobel_magnitude, cmap='gray')
ax[3].set_title("Sobel - Gradient Magnitude")
ax[3].axis('off')

ax[4].imshow(laplacian, cmap='gray')
ax[4].set_title("Laplacian (Second Derivative)")
ax[4].axis('off')

plt.tight_layout()
plt.show()

