import cv2
import matplotlib.pyplot as plt
from display.plot import show
import numpy as np
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(">>>", parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
print(sys.path)
# Load the image
image = cv2.imread('../images/ghanari.jpg')

# Check if the image was loaded correctly
if image is None:
    print("Error loading image")
else:
    # Print image shape (Height, Width, Channels)
    print("Image shape:", image.shape)
    H, W, c = image.shape

    # Extract individual RGB channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]


    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray Image shape:", gray_image.shape)


    #new_image = np.minimum(255, gray_image.astype(np.int16) + 50)
    #new_image = np.minimum(255, gray_image.astype(np.int16) + 255)
    new_image = np.full_like(gray_image, 245)
    # new_image = new_image.astype(np.uint8)

    comp_image = gray_image # all(max(0, gray_image)) 
    gmin = np.min(comp_image) 
    gmax = np.max(comp_image) 
    temp = comp_image - gmin
    trans_image = (temp/gmax)*255
    print("gmin:", gmin)
    # Display a small portion of the original image array (for example, 30x50 pixel area)
    sample_size = 30, 50
    sample_portion = image[:30, :50]
    # print("Pixel sample :\n", sample_portion)

    images = {}
    images["Original Image"] = image
    images["Gray Image"] = gray_image
    images["Image Transform"] = new_image
    images["Red Channel"] = dict(image=red_channel, cmap="Reds")
    images["Green Channel"] = dict(image=green_channel, cmap="Greens")
    images["Blue Channel"] = dict(image=blue_channel, cmap="Blues")

    show(images)


