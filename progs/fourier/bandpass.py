import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = cv2.imread('../images/sanjab.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Create a Bandpass Filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
low_cutoff, high_cutoff = 30, 100  # Define cutoff frequencies

# Create mask for bandpass
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), high_cutoff, 1, -1)  # Outer circle
cv2.circle(mask, (ccol, crow), low_cutoff, 0, -1)   # Inner circle

# Apply bandpass filter
bandpass_dft = dft_shift * mask
bandpass_image = np.fft.ifft2(np.fft.ifftshift(bandpass_dft))
bandpass_image = np.abs(bandpass_image)

# Create a Bandreject Filter
bandreject_mask = 1 - mask

# Apply bandreject filter
bandreject_dft = dft_shift * bandreject_mask
bandreject_image = np.fft.ifft2(np.fft.ifftshift(bandreject_dft))
bandreject_image = np.abs(bandreject_image)

# Visualization
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].set_title("Magnitude Spectrum")
ax[1].axis('off')

ax[2].imshow(bandpass_image, cmap='gray')
ax[2].set_title("Bandpass Filtered")
ax[2].axis('off')

ax[3].imshow(bandreject_image, cmap='gray')
ax[3].set_title("Bandreject Filtered")
ax[3].axis('off')

plt.tight_layout()
plt.show()

