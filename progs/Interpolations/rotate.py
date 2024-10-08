import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
img = cv2.imread('panda.jpg', 0)

# Print image shape (Height, Width)
print("Image shape:", img.shape)
h, w = img.shape[:2]

# Calculate the center, angle, and scale for rotation
center = (w / 2, h / 2)
angle = 5
scale = 1.0

# Get the rotation matrix
M = cv2.getRotationMatrix2D(center, angle, scale)

# Calculate new bounding dimensions for the rotated image
abs_cos = abs(M[0, 0])
abs_sin = abs(M[0, 1])
bound_w = int(h * abs_sin + w * abs_cos)
bound_h = int(h * abs_cos + w * abs_sin)

# Adjust the rotation matrix to take into account translation
M[0, 2] += bound_w / 2 - center[0]
M[1, 2] += bound_h / 2 - center[1]

# Apply the affine transformation (rotation)
rotated = cv2.warpAffine(img, M, (bound_w, bound_h))

# Convert grayscale images to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
# Display the images using matplotlib
plt.figure(figsize=(15, 10))

# Rotated Image
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Nearest Neighbor Scaling
plt.subplot(1, 2, 2)
plt.imshow(rotated_rgb)
plt.title(f"Rotated by {angle} Degrees")
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()

