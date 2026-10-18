import numpy as np
import matplotlib.pyplot as plt

# Create a black image
gray_image = np.zeros((100, 100), dtype=np.uint8)

# Make it completely white
new_image = np.full_like(gray_image, 205)

print("dtype:", new_image.dtype)
print("min:", new_image.min())
print("max:", new_image.max())

plt.imshow(new_image, cmap="gray", vmin=0, vmax=255)
plt.title("White Image")
plt.axis("off")
plt.show()
