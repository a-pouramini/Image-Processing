
import cv2
import matplotlib.pyplot as plt
# Display the images using matplotlib

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
