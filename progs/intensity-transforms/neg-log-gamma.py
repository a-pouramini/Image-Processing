import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load a grayscale image
image = cv2.imread('../images/sanjab.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Negative Transformation
negative_image = 255 - image

# 2. Log Transformation
c = 255 / np.log(1 + np.max(image))  # Scale factor to maintain intensity range
log_image = c * np.log(1 + image)
log_image = np.array(log_image, dtype=np.uint8)

# 3. Gamma Correction
gamma = 2.0  # Change this value for different gamma corrections
gamma_image = 255 * (image / 255) ** gamma
gamma_image = np.array(gamma_image, dtype=np.uint8)

# Plotting the images
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(negative_image, cmap='gray')
axs[1].set_title('Negative Transformation')
axs[1].axis('off')

axs[2].imshow(log_image, cmap='gray')
axs[2].set_title('Log Transformation')
axs[2].axis('off')

axs[3].imshow(gamma_image, cmap='gray')
axs[3].set_title('Gamma Correction (γ=2.0)')
axs[3].axis('off')

plt.tight_layout()
plt.show()

