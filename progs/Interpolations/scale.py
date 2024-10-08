import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
img = cv2.imread('ghanari.jpg')

# Print image shape (Height, Width)
print("Image shape:", img.shape)
height, width = img.shape[:2]
orig_dim = (width, height)

# Scaling Down
scale_percent = 15
width = int(width * scale_percent / 100)
height = int(height * scale_percent / 100)
dim_scale_down = (width, height)

# Resize  
scaled_down = cv2.resize(img, dim_scale_down, interpolation=cv2.INTER_NEAREST)

# Scaling
width = int(width * 100 / scale_percent)
height = int(height * 100 / scale_percent)
dim_scale_back = (width, height)

# Resize using different interpolation methods
scaled = cv2.resize(scaled_down, dim_scale_back, interpolation=cv2.INTER_NEAREST)
scaled1 = cv2.resize(scaled_down, dim_scale_back, interpolation=cv2.INTER_LINEAR)
scaled2 = cv2.resize(scaled_down, dim_scale_back, interpolation=cv2.INTER_CUBIC)

# Convert grayscale images to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scaled_down_rgb = cv2.cvtColor(scaled_down, cv2.COLOR_BGR2RGB)
scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
scaled1_rgb = cv2.cvtColor(scaled1, cv2.COLOR_BGR2RGB)
scaled2_rgb = cv2.cvtColor(scaled2, cv2.COLOR_BGR2RGB)

# Display the images using matplotlib
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title(f"Original Image {orig_dim}")
plt.axis('off')

# Scaling Down
plt.subplot(2, 3, 2)
plt.imshow(scaled_down_rgb)
plt.title(f"Scaled Down {dim_scale_down}")
plt.axis('off')

# Nearest Neighbor Scaling
plt.subplot(2, 3, 4)
plt.imshow(scaled_rgb)
plt.title(f"Nearest Neighbor Scaling {dim_scale_back}")
plt.axis('off')

# Bilinear Scaling
plt.subplot(2, 3, 5)
plt.imshow(scaled1_rgb)
plt.title(f"Bilinear Scaling {dim_scale_back}")
plt.axis('off')

# Bicubic Scaling
plt.subplot(2, 3, 6)
plt.imshow(scaled2_rgb)
plt.title(f"Bicubic Scaling {dim_scale_back}")
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()

