import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('../images/sanjab.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Calculate the histogram of the image
hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

# Step 2: Compute the cumulative distribution function (CDF)
cdf = hist.cumsum()

# Normalize the CDF to be in the range [0, 255]
cdf_normalized = cdf * 255 / cdf[-1]  # Scale the CDF to the range [0, 255]

# Step 3: Use the normalized CDF to map the pixel intensities
equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

# Reshape the flat equalized image back to its original shape
equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)

# Display the original and equalized images for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.show()

# Display the histograms for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Histogram")
plt.hist(image.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.subplot(1, 2, 2)
plt.title("Equalized Histogram")
plt.hist(equalized_image.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
plt.show()

