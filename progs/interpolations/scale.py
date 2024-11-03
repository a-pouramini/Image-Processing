import cv2
import numpy as np
import matplotlib.pyplot as plt
from display.plot import show

# Load the image in grayscale mode
image = cv2.imread('../images/ghanari.jpg')

# Print image shape (Height, Width)
height, width = image.shape[:2]

# Scaling Down
scale_percent = 20
width = int(width * scale_percent / 100)
height = int(height * scale_percent / 100)
scaled_down_dim = (width, height)

# Resize  
scaled_down = cv2.resize(image, scaled_down_dim, interpolation=cv2.INTER_NEAREST)
scaled_down_lin = cv2.resize(image, scaled_down_dim, interpolation=cv2.INTER_LINEAR)
scaled_down_cub = cv2.resize(image, scaled_down_dim, interpolation=cv2.INTER_CUBIC)

# Scaling
width = int(width * 100 / scale_percent)
height = int(height * 100 / scale_percent)
scaled_back_dim = (width, height)

# Resize using different interpolation methods
scaled = cv2.resize(scaled_down, scaled_back_dim, interpolation=cv2.INTER_NEAREST)
scaled_lin = cv2.resize(scaled_down_lin, scaled_back_dim, interpolation=cv2.INTER_LINEAR)
scaled_cub = cv2.resize(scaled_down_cub, scaled_back_dim, interpolation=cv2.INTER_CUBIC)


# Display the images using matplotlib
images = {}
images["Original Image"] = image
images["Scaled Down"] = scaled_down
images[""] = None
images["Nearest Neighbor Scaling"] = scaled
images["Bilinear Scaling"] = scaled_lin
images["Bicubic Scaling"] = scaled_cub

show(images)

