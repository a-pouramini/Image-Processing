import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def show(images, images_per_row=3, show_images=True):
    total = len(images)
    rows = total // images_per_row
    if total % images_per_row != 0: rows += 1
    plt.figure(figsize=(15, 10))

    i = 1
    for title, image in images.items():
        if image is None:
            i += 1
            continue
        plt.subplot(rows, images_per_row, i)
        kwargs = {}
        if type(image) == dict:
            kwargs = image.copy() 
            image = kwargs.pop("image")
        if len(image.shape) == 2: # Gray Image
            if not "cmap" in kwargs:
                plt.imshow(image, cmap='gray', **kwargs)
            else:
                plt.imshow(image, **kwargs)
        else: # Colored Image
            # Convert the image to RGB (OpenCV uses BGR by default)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_image, **kwargs)

        title += f" {image.shape[:2]}"
        plt.title(title)
        plt.axis('off')
        i += 1
    if show_images:
        plt.tight_layout()
        plt.show()


# Load the image in grayscale mode
image = cv2.imread('ghanari.jpg')

# Print image shape (Height, Width)
height, width = image.shape[:2]

# Scaling Down
scale_percent = 20
width = int(width * scale_percent / 100)
height = int(height * scale_percent / 100)
scaled_down_dim = (width, height)

# Resize  
scaled_down = cv2.resize(image, scaled_down_dim, interpolation=cv2.INTER_NEAREST)

# Scaling
width = int(width * 100 / scale_percent)
height = int(height * 100 / scale_percent)
scaled_back_dim = (width, height)

# Resize using different interpolation methods
scaled = cv2.resize(scaled_down, scaled_back_dim, interpolation=cv2.INTER_NEAREST)
scaled_lin = cv2.resize(scaled_down, scaled_back_dim, interpolation=cv2.INTER_LINEAR)
scaled_cub = cv2.resize(scaled_down, scaled_back_dim, interpolation=cv2.INTER_CUBIC)


# Display the images using matplotlib
images = {}
images["Original Image"] = image
images["Scaled Down"] = scaled_down
images[""] = None
images["Nearest Neighbor Scaling"] = scaled
images["Bilinear Scaling"] = scaled_lin
images["Bicubic Scaling"] = scaled_cub

show(images)

