import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack

def save_image_and_fft(image, title, output_folder, filename):
    """Saves a single image and its FFT magnitude spectrum as a combined image."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Create a combined figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title("Magnitude Spectrum")
    axes[1].axis('off')

    # Save combined image
    combined_path = os.path.join(output_folder, f"{filename}_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {combined_path}")

def generate_bar_image(size=(256, 256), bar_width=10, orientation='vertical'):
    """Generates an image with equally spaced alternating black and white bars."""
    img = np.zeros(size, dtype=np.uint8)
    if orientation == 'vertical':
        for i in range(0, size[1], 2 * bar_width):  # Step through 2*bar_width
            img[:, i:i + bar_width] = 255  # Set white bars
    else:
        for i in range(0, size[0], 2 * bar_width):  # Step through 2*bar_width
            img[i:i + bar_width, :] = 255  # Set white bars
    return img



def generate_grid_image(size=(256, 256), bar_widths=(10, 20)):
    """
    Generates an image with a grid (horizontal and vertical bars).
    Allows varied thickness for horizontal and vertical bars.

    Args:
        size (tuple): The size of the image (height, width).
        bar_widths (tuple): A tuple (vertical_bar_width, horizontal_bar_width).
    """
    vertical_bar_width, horizontal_bar_width = bar_widths
    img = np.zeros(size, dtype=np.uint8)

    # Add vertical bars
    for i in range(0, size[1], 2 * vertical_bar_width):
        img[:, i:i + vertical_bar_width] = 255

    # Add horizontal bars
    for j in range(0, size[0], 2 * horizontal_bar_width):
        img[j:j + horizontal_bar_width, :] = 255

    return img



def generate_narrow_bar_image(size=(256, 256), bar_width=10, orientation='vertical'):
    """Generates an image with bars (vertical or horizontal)."""
    img = np.zeros(size, dtype=np.uint8)
    if orientation == 'vertical':
        img[:, ::2 * bar_width] = 255  # Create vertical bars
    else:
        img[::2 * bar_width, :] = 255  # Create horizontal bars
    return img

def generate_checkerboard(size=(256, 256), block_size=32):
    """Generates a checkerboard image."""
    img = np.zeros(size, dtype=np.uint8)
    for i in range(0, size[0], block_size * 2):
        for j in range(0, size[1], block_size * 2):
            img[i:i + block_size, j:j + block_size] = 255
            img[i + block_size:i + 2 * block_size, j + block_size:j + 2 * block_size] = 255
    return img

def generate_rectangle_image(size=(256, 256), rect_size=(100, 50), rotation_angle=0):
    """Generates an image with a centered rectangle, optionally rotated."""
    img = np.zeros(size, dtype=np.uint8)
    start_x = (size[1] - rect_size[1]) // 2
    start_y = (size[0] - rect_size[0]) // 2
    img[start_y:start_y + rect_size[0], start_x:start_x + rect_size[1]] = 255
    
    if rotation_angle != 0:
        center = (size[1] // 2, size[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (size[1], size[0]))
    return img

def generate_cross_pattern(size=(256, 256), line_thickness=3):
    """
    Generates an image with a vertical and horizontal line intersecting at the center.
    The Fourier transform of this image will appear as a cross (plus) at the center.
    """
    img = np.zeros(size, dtype=np.uint8)
    center_x, center_y = size[1] // 2, size[0] // 2

    # Draw the vertical line
    img[:, center_x - line_thickness // 2 : center_x + line_thickness // 2] = 255

    # Draw the horizontal line
    img[center_y - line_thickness // 2 : center_y + line_thickness // 2, :] = 255

    return img


def generate_sinusoidal_wave(size=(256, 256), frequency=10, angle=0):
    """Generates an image with sinusoidal waves at a given frequency and angle."""
    x = np.arange(0, size[1])
    y = np.arange(0, size[0])
    xv, yv = np.meshgrid(x, y)
    theta = np.deg2rad(angle)
    wave = np.sin(2 * np.pi * frequency * (xv * np.cos(theta) + yv * np.sin(theta)) / size[1])
    wave = ((wave + 1) / 2 * 255).astype(np.uint8)  # Normalize to 0-255
    return wave

def main():
    """Main function to generate images, save them, and display their Fourier transforms."""
    output_folder = "output"

    # Original Lena Image
    lena_image = cv2.imread(cv2.samples.findFile('lena.jpg'), cv2.IMREAD_GRAYSCALE)
    if lena_image is None:
        print("Lena image not found! Using a placeholder.")
        lena_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    save_image_and_fft(lena_image, "Original Lena Image", output_folder, "lena")

    # Overlays: High Contrast Bars on Lena
    for bar_width in [5, 10, 20]:
        lena_vertical = generate_bar_image(size=lena_image.shape, bar_width=bar_width, orientation='vertical')
        lena_overlay = cv2.addWeighted(lena_image, 0.5, lena_vertical, 0.5, 0)
        save_image_and_fft(lena_overlay, f"Lena with Vertical Bars (Width {bar_width})", 
                           output_folder, f"lena_vertical_bars_{bar_width}")

        lena_horizontal = generate_bar_image(size=lena_image.shape, bar_width=bar_width, orientation='horizontal')
        lena_overlay = cv2.addWeighted(lena_image, 0.5, lena_horizontal, 0.5, 0)
        save_image_and_fft(lena_overlay, f"Lena with Horizontal Bars (Width {bar_width})", 
                           output_folder, f"lena_horizontal_bars_{bar_width}")

        lena_diagonal = generate_sinusoidal_wave(size=lena_image.shape, frequency=bar_width, angle=45)
        lena_overlay = cv2.addWeighted(lena_image, 0.5, lena_diagonal, 0.5, 0)
        save_image_and_fft(lena_overlay, f"Lena with 45-Degree Bars (Width {bar_width})", 
                           output_folder, f"lena_diagonal_bars_{bar_width}")

    # Synthetic Bars and Checkerboards
    for bar_width in [5, 10, 20]:
        bar_image_v = generate_bar_image(bar_width=bar_width, orientation='vertical')
        save_image_and_fft(bar_image_v, f"Vertical Bars (Width {bar_width})", output_folder, f"bar_vertical_{bar_width}")

        bar_image_h = generate_bar_image(bar_width=bar_width, orientation='horizontal')
        save_image_and_fft(bar_image_h, f"Horizontal Bars (Width {bar_width})", output_folder, f"bar_horizontal_{bar_width}")

    for block_size in [16, 32, 64, 128]:
        checkerboard = generate_checkerboard(block_size=block_size)
        save_image_and_fft(checkerboard, f"Checkerboard (Block {block_size})", output_folder, f"checkerboard_{block_size}")

    # Grid Experiments with Different Sizes
    for rect_size in [(10, 20), (10, 50), (10, 10)]:
        grid_image = generate_grid_image(size=(256, 256), bar_widths=rect_size)
        save_image_and_fft(grid_image, f"Grid (Size {rect_size})", output_folder, f"grid_{rect_size[0]}x{rect_size[1]}")


    # Cross Experiments with Different Sizes
    for thickness in [1,3,5,10]:
        cross_image = generate_cross_pattern(size=(256, 256), line_thickness=thickness)
        save_image_and_fft(cross_image, "Cross Pattern", output_folder, f"cross_pattern_{thickness}")    

    # Box Experiments with Different Sizes
    for rect_size in [(50, 50), (100, 50), (100, 100)]:
        box_image = generate_rectangle_image(rect_size=rect_size)
        save_image_and_fft(box_image, f"Box (Size {rect_size})", output_folder, f"box_{rect_size[0]}x{rect_size[1]}")

    # Box Experiments with Different Rotation Angles
    for angle in [45, 60, 80]: 
        rotated_box = generate_rectangle_image(rect_size=(100,50), rotation_angle=angle)
        save_image_and_fft(rotated_box, f"Rotated Box (Size {rect_size})", output_folder, f"rotated_box_{angle}")

if __name__ == "__main__":
    main()

