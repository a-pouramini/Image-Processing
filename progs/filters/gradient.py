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
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=11)  # Compute Laplacian
laplacian = cv2.convertScaleAbs(laplacian)  # Normalize for display

# Highboost Filtering
# Apply a Gaussian blur (low-pass filter)
blurred_image = cv2.GaussianBlur(image, (11, 11), 0)

# Highboost filter equation: I_highboost = I + k * (I - I_lowpass)
k = 1.5  # Boost factor
highboost_image = cv2.addWeighted(image, 1 + k, blurred_image, -k, 0)

# Visualization
fig, ax = plt.subplots(2, 4, figsize=(20, 10))

# Original Image
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title("Original Image")
ax[0, 0].axis('off')

# Sobel - Horizontal Edges
ax[0, 1].imshow(sobel_x, cmap='gray')
ax[0, 1].set_title("Sobel - Horizontal Edges")
ax[0, 1].axis('off')

# Sobel - Vertical Edges
ax[0, 2].imshow(sobel_y, cmap='gray')
ax[0, 2].set_title("Sobel - Vertical Edges")
ax[0, 2].axis('off')

# Sobel - Gradient Magnitude
ax[0, 3].imshow(sobel_magnitude, cmap='gray')
ax[0, 3].set_title("Sobel - Gradient Magnitude")
ax[0, 3].axis('off')

# Laplacian (Second Derivative)
ax[1, 0].imshow(laplacian, cmap='gray')
ax[1, 0].set_title("Laplacian (Second Derivative)")
ax[1, 0].axis('off')

# Highboost Filtered Image
ax[1, 1].imshow(highboost_image, cmap='gray')
ax[1, 1].set_title("Highboost Filtered")
ax[1, 1].axis('off')

# Displaying additional plots for comparison (You can adjust as needed)
# Here we'll display the blurred image (low-pass filter) for comparison
ax[1, 2].imshow(blurred_image, cmap='gray')
ax[1, 2].set_title("Blurred Image (Low-pass Filter)")
ax[1, 2].axis('off')

# Empty plot space for flexibility
ax[1, 3].axis('off')

plt.tight_layout()
plt.show()

