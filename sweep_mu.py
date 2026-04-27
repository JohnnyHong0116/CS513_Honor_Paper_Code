# import necessary libraries
import numpy as np
from scipy.signal import convolve2d
from skimage import data, transform, img_as_float
from PIL import Image, ImageDraw, ImageFont
import cv2

# Setup parameters for the experiment
IMAGE_TYPE = "camera"  # Change to "text", "camera", "aruco"
IMAGE_SIZE = 64
BLUR_RADIUS = 2
NOISE_SIGMA = 0.005
ARUCO_ID = 0
TEXT_STRING = "ABC"
RANDOM_SEED = 1

# Test different Tikhonov mu values
MU_VALUES = [
    0.001, 0.002, 0.003, 0.004, 0.005,
    0.0075, 0.01, 0.015, 0.02, 0.03,
    0.04, 0.05, 0.075, 0.1, 0.15, 0.2
]

# Relative error: ||x_true - x_est|| / ||x_true||
def relative_error(x_true, x_est):
    return np.linalg.norm(x_true - x_est) / np.linalg.norm(x_true)

# Load Image from skimage for camera image and resize it to the desired size
def load_camera_image(size):
    img = img_as_float(data.camera())
    # Resize the image to the desired size using anti-aliasing to prevent artifacts
    img_small = transform.resize(img, (size, size), anti_aliasing=True)
    # Clip values to [0, 1] => normalize to [0, 1]
    return np.clip(img_small, 0, 1)

# Create a simple "ABC" text image with PIL and resize it to the desired size
def load_text_image(size, text):
    canvas_size = 256
    # Draw black text on a white canvas
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    # Load arial font
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except OSError:
        font = ImageFont.load_default()

    # Center the text on the canvas
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2
    y = (canvas_size - text_h) // 2

    # Draw the text in black (0) on white (255) background
    draw.text((x, y), text, fill=0, font=font)

    # Convert the PIL image to a numpy array
    img_np = np.array(img, dtype=np.float64) / 255.0
    # Resize to desired size using anti-aliasing to prevent artifacts
    img_small = transform.resize(img_np, (size, size), anti_aliasing=True)
    # Clip values to [0, 1] => normalize to [0, 1]
    return np.clip(img_small, 0, 1)

# Create an ArUco marker image with OpenCV and resize it to the desired size
def load_aruco_image(size, marker_id):
    # Generate a 4x4 ArUco marker with the chosen ID
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Create a marker image of size 200x200 pixels
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)

    # Place the marker in the center of a white canvas
    canvas = 255 * np.ones((256, 256), dtype=np.uint8)
    canvas[28:228, 28:228] = marker

    # Convert the image to float
    img = canvas.astype(np.float64) / 255.0
    # Resize to desired size using anti-aliasing to prevent artifacts
    img_small = transform.resize(img, (size, size), anti_aliasing=True)
    # Clip values to [0, 1] => normalize to [0, 1]
    return np.clip(img_small, 0, 1)

# Build a circular average blur kernel with the given radius
def create_circular_average_blur_kernel(radius):
    # The condition x^2 + y^2 <= radius^2 selects pixels inside the circle
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    # Create a kernel of size (2*radius+1, 2*radius+1) and set the pixels inside the circle to 1
    kernel = np.zeros((2*radius+1, 2*radius+1))
    kernel[mask] = 1.0

    # Normalize the kernel so all entries sum to 1
    kernel = kernel / np.sum(kernel)
    return kernel

# Blur model
def blur_image(img, kernel):
    # Convolution applies the blur kernel to the image
    return convolve2d(img, kernel, mode="same", boundary="symm")

# Build the blur matrix A for the forward model b = Ax
def build_blur_matrix(shape, kernel):
    h, w = shape
    n = h * w  # number of pixels in the image
    A = np.zeros((n, n))

    for j in range(n):
        # Put a 1 in pixel j and 0 elsewhere to create a basis image
        # Blurring this basis image gives column j of A
        basis = np.zeros((h, w))
        basis.flat[j] = 1.0
        blurred_basis = blur_image(basis, kernel)
        A[:, j] = blurred_basis.flatten()

    return A

# Load the original image
if IMAGE_TYPE == "camera":
    x_img = load_camera_image(IMAGE_SIZE)
elif IMAGE_TYPE == "text":
    x_img = load_text_image(IMAGE_SIZE, TEXT_STRING)
elif IMAGE_TYPE == "aruco":
    x_img = load_aruco_image(IMAGE_SIZE, ARUCO_ID)
else:
    raise ValueError("IMAGE_TYPE must be 'text', 'camera', or 'aruco'")

# Flatten the image into a vector for linear algebra operations
x_vec = x_img.flatten()

# Build the blur matrix A for the forward model b = Ax
kernel = create_circular_average_blur_kernel(BLUR_RADIUS)
A = build_blur_matrix(x_img.shape, kernel)

# Make blurred data Ax, then add noise to form b
rng = np.random.default_rng(RANDOM_SEED)
# A is the blur matrix and x is the original image vector
b_blur = A @ x_vec
# Reshape back into image form for adding noise and visualization
b_img = b_blur.reshape(x_img.shape)

# Add random noise to model measurement error
# b_noisy = Ax + noise
b_noisy_img = b_img + rng.normal(0, NOISE_SIGMA, x_img.shape)
# Clip values to [0, 1] so the noisy image remains in a valid display/intensity range
b_noisy_img = np.clip(b_noisy_img, 0, 1)
# Flatten the blurred noisy image into a vector for solving linear systems
b_vec = b_noisy_img.flatten()

# Precompute A^T A and A^T b for Tikhonov regularization
AT = A.T
ATA = AT @ A
ATb = AT @ b_vec

# Test different Tikhonov mu values
print("Testing Tikhonov mu values...")
mu_results = []

for mu in MU_VALUES:
    # Tikhonov regularization:
    # solve min ||Ax - b||^2 + mu^2 ||x||^2
    # The normal equations are (A^T A + mu^2 I) x = A^T b
    x_tikh = np.linalg.solve(ATA + mu**2 * np.eye(A.shape[1]), ATb)
    x_tikh_img = x_tikh.reshape(x_img.shape)

    # Compute relative error for this mu
    err = relative_error(x_img, x_tikh_img)
    mu_results.append((mu, err))

# Find best mu with lowest relative error
best_mu, best_err = min(mu_results, key=lambda item: item[1])

# Print result summary
print("Best Tikhonov mu among tested values: {:.8f}".format(best_mu))
print("Best Tikhonov relative error: {:.8f}".format(best_err))