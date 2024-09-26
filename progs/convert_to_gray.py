import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to manually convert RGB image to grayscale using weighted average
def rgb_to_gray_manual(image):
    return (0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]).astype(np.uint8)

# Load the original image
image = cv2.imread('sanjab.jpg')

if image is None:
    print("Error loading image")
else:
    # Convert the image to grayscale using OpenCV
    gray_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert original image from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale manually using weighted averaging
    gray_manual = rgb_to_gray_manual(image)

    # Convert grayscale image to color using Pseudocoloring (False Coloring)
    colored_jet = cv2.applyColorMap(gray_opencv, cv2.COLORMAP_JET)
    colored_rainbow = cv2.applyColorMap(gray_manual, cv2.COLORMAP_RAINBOW)

    gray_3_channel_1 = cv2.cvtColor(gray_opencv,cv2.COLOR_GRAY2RGB)
    #gray_3_channel_2 = cv2.merge([gray_opencv,gray_opencv,gray_opencv])
    
    # Plot all the images
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Grayscale (OpenCV)
    plt.subplot(2, 3, 2)
    plt.imshow(gray_opencv, cmap='gray')
    plt.title("Grayscale (OpenCV)")
    plt.axis('off')

    # Grayscale (Manual)
    plt.subplot(2, 3, 3)
    plt.imshow(gray_manual, cmap='gray')
    plt.title("Grayscale (Manual)")
    plt.axis('off')

    # Pseudocolored Image (OpenCV Grayscale)
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(gray_3_channel_1, cv2.COLOR_BGR2RGB))
    plt.title("Grayscale (3 Channel)")
    plt.axis('off')

    # Pseudocolored Image (OpenCV Grayscale)
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(colored_jet, cv2.COLOR_BGR2RGB))
    plt.title("Pseudocolored (COLORMAP_JET)")
    plt.axis('off')

    # Pseudocolored Image (Manual Grayscale)
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(colored_rainbow, cv2.COLOR_BGR2RGB))
    plt.title("Pseudocolored (COLORMAP_RAINBOW)")
    plt.axis('off')

    # Show the plot
    # plt.tight_layout()
    plt.show()

