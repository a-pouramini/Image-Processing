import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('gol.jpg')

# Check if the image was loaded correctly
if image is None:
    print("Error loading image")
else:
    # Print image shape (Height, Width, Channels)
    print("Image shape:", image.shape)
    M, N, c = image.shape

    # Extract individual RGB channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray Image shape:", gray_image.shape)

    # Display a small portion of the original image array (for example, 5x5 pixel area)
    pixel_sample = image[M-10:M-1, N-10:N-1]
    print("Pixel sample (3x3 portion):\n", pixel_sample)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the images using matplotlib
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title(f"Original Image\nShape: {image.shape}")
    plt.axis('off')

    # Grayscale Image
    plt.subplot(2, 3, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"Grayscale Image\nShape: {gray_image.shape}")
    plt.axis('off')

    # Blue Channel
    plt.subplot(2, 3, 4)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title("Blue Channel")
    plt.axis('off')

    # Green Channel
    plt.subplot(2, 3, 5)
    plt.imshow(green_channel, cmap='Greens')
    plt.title("Green Channel")
    plt.axis('off')

    # Red Channel
    plt.subplot(2, 3, 6)
    plt.imshow(red_channel, cmap='Reds')
    plt.title("Red Channel")
    plt.axis('off')

    # Display the pixel values as text
    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.title("Pixel Sample (3x3) Values")
    plt.imshow(pixel_sample)
    
    #pixel_text = ""
    #for row in pixel_sample:
    #    for pixel in row:
    #        pixel_text += str(pixel) + ""  # Print each pixel value (B, G, R)
    #    pixel_text += "\n"
    
    #plt.text(0.5, 0.5, pixel_text, fontsize=12, ha='center', va='center', family='monospace')

    # Show the images
    plt.tight_layout()
    plt.show()

