# import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data, transform, img_as_float
from PIL import Image, ImageDraw, ImageFont
import cv2

# Setup parameters for the experiment
IMAGE_TYPE = "aruco"  # Change to "text", "camera", "aruco"
IMAGE_SIZE = 64 # If larger than 64, takes much longer to run and may cause memory issues
BLUR_RADIUS = 2 # Increase to add more blur (more ill-conditioned)
NOISE_SIGMA = 0.005 # Increase to add more noise to the blurred image (more measurement error)
# Best settings for TSVD_K & TIKHONOV_MU can be found in README.md
TSVD_K = 3300
TIKHONOV_MU = 0.05
ARUCO_ID = 0
TEXT_STRING = "ABC"
OUTPUT_DIR = "outputs"
RANDOM_SEED = 1

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

# Build the blur matrix A for the model b = Ax
def build_blur_matrix(shape, kernel):
    h, w = shape
    n = h * w # number of pixels in the image
    A = np.zeros((n, n)) 

    for j in range(n):
        # Put a 1 in pixel j and 0 elsewhere to create a basis image
        # Blurring this basis image gives column j of A
        basis = np.zeros((h, w))
        basis.flat[j] = 1.0
        blurred_basis = blur_image(basis, kernel)
        A[:, j] = blurred_basis.flatten()

    return A

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Build the blur matrix A for the model b = Ax
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

# Direct solve
print("Direct solving Ax = b...")
x_direct = np.linalg.solve(A, b_vec)

# Truncated SVD solve
print("Computing TSVD...")
U, s, Vt = np.linalg.svd(A, full_matrices=False)
k = min(TSVD_K, len(s))
# Keep only the first k singular values and corresponding singular vectors
S_inv = np.diag(1.0 / s[:k])
# x_tsvd = V_k Sigma_k^{-1} U_k^T b, U_k and V_k are the first k columns of U and V
x_tsvd = Vt[:k, :].T @ S_inv @ U[:, :k].T @ b_vec

# Condition number = largest singular value / smallest singular value
cond_A = s[0] / s[-1]

# Tikhonov regularization
print("Computing Tikhonov regularization...")
# Solve for min ||Ax - b||^2 + mu^2 ||x||^2
# The normal equations are (A^T A + mu^2 I) x = A^T b
mu = TIKHONOV_MU
AT = A.T
ATA = AT @ A
# Solve the regularized normal equations for x
x_tikh = np.linalg.solve(ATA + mu**2 * np.eye(A.shape[1]), AT @ b_vec)

# Reshape all solutions back into images
x_direct_img = x_direct.reshape(x_img.shape)
x_tsvd_img = x_tsvd.reshape(x_img.shape)
x_tikh_img = x_tikh.reshape(x_img.shape)

# Metrics for relative error compared to the original image
blur_err = relative_error(x_img, b_noisy_img)
direct_err = relative_error(x_img, x_direct_img)
tsvd_err = relative_error(x_img, x_tsvd_img)
tikh_err = relative_error(x_img, x_tikh_img)

# Save metrics and parameters to a text file
metrics_lines = [
    "Image type: {}".format(IMAGE_TYPE)
]
if IMAGE_TYPE == "text":
    metrics_lines.append("Text string: {}".format(TEXT_STRING))
if IMAGE_TYPE == "aruco":
    metrics_lines.append("ArUco ID: {}".format(ARUCO_ID))
metrics_lines.append("Blur radius: {}".format(BLUR_RADIUS))
metrics_lines.append("Noise sigma: {}".format(NOISE_SIGMA))
metrics_lines.append("Condition number of A: {:.6e}".format(cond_A))
metrics_lines.append("Blurred relative error: {:.8f}".format(blur_err))
metrics_lines.append("Direct solve relative error: {:.8f}".format(direct_err))
metrics_lines.append("TSVD relative error: {:.8f}".format(tsvd_err))
metrics_lines.append("Tikhonov relative error: {:.8f}".format(tikh_err))
metrics_filename = "{}_metrics.txt".format(IMAGE_TYPE)

with open(os.path.join(OUTPUT_DIR, metrics_filename), "w") as f:
    f.write("\n".join(metrics_lines))
    f.write("\n")

# Plot image results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Show the original image
axes[0, 0].imshow(x_img, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

# Show the blurred + noisy image
axes[0, 1].imshow(b_noisy_img, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title("Blurred + noisy")
axes[0, 1].axis("off")

# Show the direct solve image
axes[0, 2].imshow(np.clip(x_direct_img, 0, 1), cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title("Direct solve")
axes[0, 2].axis("off")

# Show the TSVD reconstruction
axes[1, 0].imshow(np.clip(x_tsvd_img, 0, 1), cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("TSVD, k={}".format(k))
axes[1, 0].axis("off")

# Show the Tikhonov reconstruction
axes[1, 1].imshow(np.clip(x_tikh_img, 0, 1), cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title("Tikhonov, mu={}".format(mu))
axes[1, 1].axis("off")

axes[1, 2].axis("off")

plt.tight_layout()
# Save the figure to the output directory
plt.savefig(os.path.join(OUTPUT_DIR, "{}_reconstructions.png".format(IMAGE_TYPE)), dpi=200)
plt.close()

# Plot singular values
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# Use a semilog plot to show the singular values
ax.semilogy(s, "-o", markersize=2)
ax.set_title("Singular Value Decay of the Blur Matrix A")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value")
ax.grid(True, alpha=0.3)
plt.tight_layout()
# Save the figure to the output directory
plt.savefig(os.path.join(OUTPUT_DIR, "{}_singular_values.png".format(IMAGE_TYPE)), dpi=200)
plt.close()

# Print summary of saved files
print("\nSaved files to '{}'".format(OUTPUT_DIR))
print("- {}".format(metrics_filename))
print("- {}_reconstructions.png".format(IMAGE_TYPE))
print("- {}_singular_values.png".format(IMAGE_TYPE))
