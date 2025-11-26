######   LAB 1

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1: Image Acquisition ---
try:
    img_path = "/content/dog.jpeg"
    og_img = cv2.imread(img_path)

    if og_img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    og_img_rgb = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)

    print("Image acquired successfully!")
    print("Original image shape:", og_img_rgb.shape)

except Exception as e:
    print(f"Error loading image: {e}")
    og_img_rgb = None

# --- Step 2: Grayscale Conversion ---
if og_img_rgb is not None:
    grayscale_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale successfully!")
    print("Grayscale image shape:", grayscale_img.shape)

    # --- Step 3: Image Sampling (Resizing) ---
    new_size = (250, 120)
    resized_color_img = cv2.resize(og_img_rgb, new_size)
    resized_grayscale_img = cv2.resize(grayscale_img, new_size)

    # Display original and resized images
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(og_img_rgb)
    plt.title('Original Color Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(grayscale_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(resized_color_img)
    plt.title('Resized Color Image (250x120)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(resized_grayscale_img, cmap='gray')
    plt.title('Resized Grayscale Image (250x120)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Step 4: Convert Grayscale back to 3-Channel (for comparison) ---
    reverted_color_image = cv2.cvtColor(resized_grayscale_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(resized_grayscale_img, cmap='gray')
    plt.title('Input: Resized Grayscale (1-Channel)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reverted_color_image)
    plt.title('Output: Converted to RGB (3-Channel)')
    plt.axis('off')
    plt.show()

    # --- Step 5: Intensity Profile ---
    row_index = resized_grayscale_img.shape[0] // 2
    grayscale_intensity = resized_grayscale_img[row_index, :]
    red_channel = resized_color_img[row_index, :, 0]
    green_channel = resized_color_img[row_index, :, 1]
    blue_channel = resized_color_img[row_index, :, 2]

    plt.figure(figsize=(12, 6))
    plt.plot(grayscale_intensity, color='black', label='Grayscale Intensity')
    plt.plot(red_channel, color='red', label='Red Channel Intensity')
    plt.plot(green_channel, color='green', label='Green Channel Intensity')
    plt.plot(blue_channel, color='blue', label='Blue Channel Intensity')
    plt.title(f'Intensity Profile at Row {row_index}')
    plt.xlabel('Pixel Position (Column)')
    plt.ylabel('Intensity Value (0-255)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Addition 1: Histogram of Grayscale ---
    plt.figure(figsize=(10, 6))
    plt.hist(grayscale_img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.8)
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity (0=Black, 255=White)")
    plt.ylabel("Number of Pixels")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # --- Addition 2: 3D RGB Color Space Plot ---
    pixels = og_img_rgb.reshape(-1, 3)
    if len(pixels) > 10000:  # sample for performance
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]

    pixel_colors = pixels / 255.0
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, c=pixel_colors, marker='o', alpha=0.6)
    ax.set_xlabel("Red Channel")
    ax.set_ylabel("Green Channel")
    ax.set_zlabel("Blue Channel")
    plt.title("3D Scatter Plot of Pixel Colors in RGB Space")
    plt.show()


###  LAB 2


import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files


# Read the image from the specified path
img = cv2.imread('/content/dog.jpeg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or could not be loaded. Please check the path or upload the image.")
    uploaded = files.upload()
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))
        img = cv2.imread(fn)
        if img is not None:
            print(f"Successfully loaded uploaded image: {fn}")
            break
    if img is None:
        print("Error: No valid image was uploaded or found. Exiting.")
    else:
        # Convert the original image (BGR format) to Grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert the original image to HSV (Hue, Saturation, Value) color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Convert the original image to LAB (Lightness, A-channel, B- channel) color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Create a figure window with size 12x6 inches
        plt.figure(figsize=(12, 6))


        # Display the original image (convert BGR to RGB for correct display in matplotlib)
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')	# Hide axis borders

        # Display the grayscale image
        plt.subplot(1, 4, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        # Display the HSV image
        plt.subplot(1, 4, 3)
        plt.imshow(hsv_img)
        plt.title('HSV Image')
        plt.axis('off')

        # Display the LAB image
        plt.subplot(1, 4, 4)
        plt.imshow(lab_img)
        plt.title('LAB Image')
        plt.axis('off')

        # Adjust layout to avoid overlapping
        plt.tight_layout()

        # Show the final output
        plt.show()
else:
    # Convert the original image (BGR format) to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the original image to HSV (Hue, Saturation, Value) color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert the original image to LAB (Lightness, A-channel, B- channel) color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Create a figure window with size 12x6 inches
    plt.figure(figsize=(12, 6))


    # Display the original image (convert BGR to RGB for correct display in matplotlib)
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')	# Hide axis borders

    # Display the grayscale image
    plt.subplot(1, 4, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Display the HSV image
    plt.subplot(1, 4, 3)
    plt.imshow(hsv_img)
    plt.title('HSV Image')
    plt.axis('off')

    # Display the LAB image
    plt.subplot(1, 4, 4)
    plt.imshow(lab_img)
    plt.title('LAB Image')
    plt.axis('off')

    # Adjust layout to avoid overlapping
    plt.tight_layout()

    # Show the final output
    plt.show()




    ### LAB 3

import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# Read the image
img = cv2.imread('/content/dog.jpeg')

# Convert BGR → RGB for proper display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Gaussian Filter (smooths image, reduces noise, but also blurs edges)
gaussian = cv2.GaussianBlur(img_rgb, (15, 15), 0)

# 2. Median Filter (very good for removing salt-pepper noise)
median = cv2.medianBlur(img_rgb, 7)

# 3. Bilateral Filter (smooths image while preserving edges clearly)
bilateral = cv2.bilateralFilter(img_rgb, 15, 75, 75)

# Display all images
plt.figure(figsize=(12, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Gaussian Filter
plt.subplot(2, 2, 2)
plt.imshow(gaussian)
plt.title("Gaussian Filter")
plt.axis('off')

# Median Filter
plt.subplot(2, 2, 3)
plt.imshow(median)
plt.title("Median Filter")
plt.axis('off')

# Bilateral Filter
plt.subplot(2, 2, 4)
plt.imshow(bilateral)
plt.title("Bilateral Filter")
plt.axis('off')

plt.tight_layout()
plt.show()



#### LAB 4

import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.colab import files

# Read the image
img = cv2.imread('/content/dog.jpeg')
if img is None:
    print("Error: Image not loaded. Please check the file path.")

# Convert to grayscale and blur
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding (Otsu + Binary Inverse)
if img is not None:
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image (Otsu + Binary Inverse)')
    plt.axis('off')
    plt.show()

# Morphological closing
kernel = np.ones((9, 9), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

# Dilate to extract background
bg = cv2.dilate(closing, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

result = np.zeros(gray.shape, dtype=np.uint8)

for contour in contours:
    if cv2.contourArea(contour) > 1000:
        cv2.fillPoly(result, [contour], 255)

# Opening to remove noise
kernel_open = np.ones((6, 6), np.uint8)
opened_result = cv2.morphologyEx(
    result, cv2.MORPH_OPEN, kernel_open, iterations=2
)

# Erode to clean edges
kernel_erode = np.ones((9, 9), np.uint8)
final_result = cv2.erode(opened_result, kernel_erode, iterations=2)

# Display final output
plt.imshow(final_result, cmap='gray')
plt.title('Final Segmented Output')
plt.axis('off')
plt.show()


### LAB 5

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 9)

def show(title, img, cmap=None):
    """Show image using matplotlib. Accepts BGR or single-channel arrays."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        # convert BGR (opencv) to RGB (for display)
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display = img
    plt.figure()
    plt.imshow(display, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def imread_safe(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load '{path}'. Put the image in working directory.")
    return img

# --- load image (BGR) ---
img_bgr = imread_safe('/content/dog.jpeg')   # replace file name if needed
h, w, _ = img_bgr.shape
print(f"Image shape (H,W,C): {img_bgr.shape}")

# Quick display
show("Original (RGB)", img_bgr)

# -------------------------------------------------------
# 1) Color space conversions
# -------------------------------------------------------

img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

show("RGB", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
show("HSV (visualized by converting back to RGB)",
     cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))
show("LAB (L channel shown)", img_lab[:, :, 0], cmap='gray')

# -------------------------------------------------------
# 2) Gamma correction (gamma < 1 brighten, gamma > 1 darken)
# -------------------------------------------------------

def gamma_correction(bgr, gamma):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(bgr, table)

gamma_08 = gamma_correction(img_bgr, 0.8)     # brighten slightly
gamma_12 = gamma_correction(img_bgr, 1.2)     # darken slightly

show("Gamma 0.8 (brighter)", gamma_08)
show("Gamma 1.2 (darker)", gamma_12)

# -------------------------------------------------------
# 3) Histogram equalization (global on luminance)
# -------------------------------------------------------

ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb)
y_eq = cv2.equalizeHist(y)
ycrcb_eq = cv2.merge((y_eq, cr, cb))
img_eq_global = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

show("Histogram Equalized (global on Y channel)", img_eq_global)

# -------------------------------------------------------
# 4) Color segmentation (red detection demo)
# -------------------------------------------------------

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask = mask1 | mask2

segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
show("Segmented red regions", segmented)

# -------------------------------------------------------
# 8) Color smoothing and sharpening
# -------------------------------------------------------

# Bilateral Filter (edge-preserving smoothing)
bilat = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
show("Bilateral Filter (color-preserving smoothing)", bilat)

# Sharpening using Unsharp Mask
def unsharp_mask_color(bgr, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(bgr, kernel_size, sigma)
    sharpened = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return sharpened

sharpened = unsharp_mask_color(img_bgr, kernel_size=(5,5), sigma=1.0, amount=0.8)
show("Sharpened (Unsharp Mask)", sharpened)


lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)
img_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

blend = cv2.addWeighted(img_bgr, 0.6, img_clahe, 0.4, 0)
show("Blend: original (0.6) + CLAHE (0.4)", blend)

cv2.imwrite('img_gamma08.png', gamma_08[:, :, ::-1])  
cv2.imwrite('img_clahe.png', img_clahe)
cv2.imwrite('img_quantized.png', img_clahe)  

print("Saved outputs: img_gamma08.png, img_clahe.png, img_quantized.png (in working directory).")
print("Done. Techniques applied: color conversions, gamma, equalization, CLAHE, white balance, segmentation, quantization, smoothing & sharpening.")



### LAB 6

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Load Left and Right Images
# -------------------------------
left = cv2.imread("/content/dog.jpeg")
right = cv2.imread("/content/dog.jpeg")

# Check if images loaded
if left is None or right is None:
    print("Error: Check image paths!")

    exit()

left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------
# PART 1: Harris Corner Detection
# -----------------------------------------------------
def harris_corner(img_gray):
    img_float = np.float32(img_gray)
    harris = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    # Corrected typo: cvv2.cvtColor -> cv2.cvtColor
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_color[harris > 0.01 * harris.max()] = [0, 0, 255]
    return img_color

harris_left = harris_corner(left_gray)
harris_right = harris_corner(right_gray)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(harris_left); plt.title("Harris Corners (Left)")
plt.subplot(1, 2, 2); plt.imshow(harris_right); plt.title("Harris Corners (Right)")
plt.show()

# -----------------------------------------------------
# PART 2: SIFT Feature Detection + Matching
# -----------------------------------------------------
sift = cv2.SIFT_create()

keyL, descL = sift.detectAndCompute(left_gray, None)
keyR, descR = sift.detectAndCompute(right_gray, None)

# FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descL, descR, k=2)

# Lowe ratio test
good_matches = []
ptsL = []
ptsR = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        ptsL.append(keyL[m.queryIdx].pt)
        ptsR.append(keyR[m.trainIdx].pt)

ptsL = np.int32(ptsL)
ptsR = np.int32(ptsR)

# Draw Matches
match_img = cv2.drawMatches(left, keyL, right, keyR, good_matches, None)

plt.figure(figsize=(15, 8))
plt.imshow(match_img)
plt.title("SIFT Feature Matching")
plt.show()



## LAB 7

#pip install ultralytics Pillow
print("ultralytics and Pillow libraries installed successfully.")
get_ipython().system('pip install ultralytics Pillow')
print("ultralytics and Pillow libraries installed successfully.")
from ultralytics import YOLO
# Load a pre-trained YOLOv5s model
model = YOLO('yolov5s.pt')
print("YOLOv5s model loaded successfully.")
from google.colab import files
from PIL import Image
import io
print("Please upload your image file. Ensure it's a valid image format (e.g., JPG, PNG).")
uploaded = files.upload()
if not uploaded:
    print("No file was uploaded. Please try again.")
else:
  for filename in uploaded.keys():
        print(f'Uploaded file: {filename}')
        try:
            image = Image.open(io.BytesIO(uploaded[filename]))
            print("Image uploaded and opened successfully. The YOLO model will handle further preprocessing.")
            
        except Exception as e:
            print(f"Error opening image file {filename}: {e}")
            image = None # Set image to None if there's an error
            import matplotlib.pyplot as plt
from IPython.display import display
if image is None:
    print("No image loaded. Please upload an image first.")
else:
    # Perform inference
    # The YOLO model can take a PIL Image directly
    results = model.predict(source=image, save=True, save_txt=True, save_conf=True, conf=0.25)
    if results and len(results) > 0:
        # The results object directly contains the annotated image in array format
        # We can convert it back to PIL Image for consistent display
        annotated_image_array = results[0].plot() # This returns an annotated numpy array
        annotated_pil_image = Image.fromarray(annotated_image_array[..., ::-1]) # Convert BGR to RGB if needed
        print("Displaying detection results:")
        display(annotated_pil_image)
        # Optionally, you can also print detection details if needed
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls # class
                conf = box.conf # confidence score
                xyxy = box.xyxy # box coordinates
                name = model.names[int(c)] # class name
                print(f"Detected: {name} with confidence {conf.item():.2f} at coordinates {xyxy.tolist()}")
    else:
        print("No detections found or an issue occurred during inference.")
        from google.colab import files
#from PIL import Image
import io
print("Please upload your image file. Ensure it's a valid image format (e.g., JPG, PNG).")
uploaded = files.upload()
if not uploaded:
    print("No file was uploaded. Please try again.")
else:
    for filename in uploaded.keys():
        print(f'Uploaded file: {filename}')
        try:
            image = Image.open(io.BytesIO(uploaded[filename]))
            print("Image uploaded and opened successfully. The YOLO model will handle further preprocessing.")
        except Exception as e:
            print(f"Error opening image file {filename}: {e}")
            import matplotlib.pyplot as plt
from IPython.display import display
if image is None:
    print("No image loaded. Please upload an image first.")
else:
    # Perform inference
    # The YOLO model can take a PIL Image directly
    results = model.predict(source=image, save=True, save_txt=True, save_conf=True, conf=0.25)
    if results and len(results) > 0:
        # The results object directly contains the annotated image in array format
        # We can convert it back to PIL Image for consistent display
        annotated_image_array = results[0].plot() # This returns an annotated numpy array
        annotated_pil_image = Image.fromarray(annotated_image_array[..., ::-1]) # Convert BGR to RGB if needed
        print("Displaying detection results:")
        display(annotated_pil_image)
        # Optionally, you can also print detection details if needed
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls # class
                conf = box.conf # confidence score
                xyxy = box.xyxy # box coordinates
                name = model.names[int(c)] # class name
                print(f"Detected: {name} with confidence {conf.item():.2f} at coordinates {xyxy.tolist()}")
    else:
        print("No detections found or an issue occurred during inference.")




        #### LAB 8


import os, time, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
import cv2

# Utilities
def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def draw_boxes_on_image_cv2(img_bgr, boxes, labels=None, scores=None, score_thresh=0.3, class_names=None, color=(0,255,0)):
    img = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box)
        sc = float(scores[i]) if scores is not None else None
        if sc is not None and sc < score_thresh:
            continue
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label_text = ""
        if labels is not None:
            lab = int(labels[i])
            if class_names and isinstance(class_names, (list,tuple)) and lab < len(class_names):
                label_text = class_names[lab]
            else:
                label_text = str(lab)
        if sc is not None:
            label_text = f"{label_text} {sc:.2f}"
        cv2.putText(img, label_text, (x1, max(15,y1-5)), font, 0.6, color, 2, cv2.LINE_AA)
    return img

# COCO labels for Faster R-CNN display
COCO_LABELS = [
    '__background__', 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]
def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader, trainset.classes

def build_resnet18_for_cifar(num_classes=10, pretrained=True, device="cpu"):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
def train_resnet(model, trainloader, testloader, device, epochs=10, lr=1e-3, out_path="resnet18_cifar10.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    history = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    best_val_acc = 0.0
    ensure_dir(out_path)

    for epoch in range(epochs):
        model.train()
        running_loss=0.; running_correct=0; running_total=0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for images, labels in loop:
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            running_correct += (preds == labels).sum().item()
            running_total += images.size(0)
        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total

        # validation
        model.eval()
        val_loss=0.; val_correct=0; val_total=0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device); labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        tqdm.write(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            tqdm.write(f"Saved best model to {out_path} (val_acc={best_val_acc:.4f})")
    return history, best_val_acc

def plot_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="train_acc"); plt.plot(history["val_acc"], label="val_acc"); plt.legend(); plt.title("Accuracy")
    plt.show()

def evaluate_classifier(model, dataloader, device, class_names=None):
    model.eval()
    all_preds=[]; all_labels=[]
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval"):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu().numpy()); all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    n = len(class_names) if class_names else int(all_labels.max()+1)
    cm = np.zeros((n,n), dtype=int)
    for t,p in zip(all_labels, all_preds): cm[t,p]+=1
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
    overall = (all_preds==all_labels).mean()
    print("Overall acc:", overall)
    if class_names:
        for i,name in enumerate(class_names): print(f"{name}: {per_class_acc[i]:.4f} (n={cm.sum(axis=1)[i]})")
    plt.figure(figsize=(7,6)); plt.imshow(cm, cmap='Blues'); plt.colorbar()
    if class_names:
        plt.xticks(np.arange(len(class_names)), class_names, rotation=90); plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.show()
    return overall, per_class_acc, cm

def load_fasterrcnn(device="cpu", pretrained=True):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None)
    model.to(device).eval()
    return model

def run_detector_on_image(model, image_path, device="cpu", conf_thresh=0.5, max_boxes=50):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError("Image not found: " + image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(rgb).to(device)
    with torch.no_grad():
        out = model([tensor])[0]
    boxes = out['boxes'].cpu().numpy()
    scores = out['scores'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    keep = scores >= conf_thresh
    boxes = boxes[keep][:max_boxes]; scores = scores[keep][:max_boxes]; labels = labels[keep][:max_boxes]
    return bgr, boxes, labels, scores

def visualize_and_save_detections(image_bgr, boxes, labels, scores, out_path="detections.jpg", score_thresh=0.4):
    img_out = draw_boxes_on_image_cv2(image_bgr, boxes, labels, scores, score_thresh, COCO_LABELS)
    cv2.imwrite(out_path, img_out)
    # show in notebook
    img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,10)); plt.imshow(img_rgb); plt.axis('off'); plt.show()
    print("Saved:", out_path)


# Choose device and seed
seed_everything(0)
device = get_device()
print("Device:", device)

trainloader, testloader, class_names = get_cifar10_dataloaders(batch_size=128, num_workers=2)
model = build_resnet18_for_cifar(num_classes=len(class_names), pretrained=True, device=device)
history, best = train_resnet(model, trainloader, testloader, device, epochs=3, lr=1e-3, out_path="resnet18_cifar10_demo.pth")
plot_history(history)
print("Best val acc (demo):", best)

device = get_device()
trainloader, testloader, class_names = get_cifar10_dataloaders(batch_size=128, num_workers=2)
model = build_resnet18_for_cifar(num_classes=len(class_names), pretrained=True, device=device)
model.load_state_dict(torch.load("resnet18_cifar10_demo.pth", map_location=device))
evaluate_classifier(model, testloader, device, class_names=class_names)

# Run this cell to upload an image file interactively in Colab
from google.colab import files
uploaded = files.upload()  # click to upload image(s) from your machine

# Suppose you uploaded 'sample.jpg', set image_path accordingly:
image_path = list(uploaded.keys())[0]
print("Uploaded:", image_path)

device = get_device()
det_model = load_fasterrcnn(device=device, pretrained=True)
bgr, boxes, labels, scores = run_detector_on_image(det_model, image_path, device=device, conf_thresh=0.45)
visualize_and_save_detections(bgr, boxes, labels, scores, out_path="detections_out.jpg", score_thresh=0.45)



