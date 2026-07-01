import cv2
import os
import numpy as np
import random

image_path = r"D:\university_tokyo\visual_media\segment-anything-main\assets\seagrass-marine-biodiversity.jpg"
mask_dir = r"D:\university_tokyo\visual_media\segment-anything-main\outputs\seagrass-marine-biodiversity\seagrass-marine-biodiversity"
output_path = r"D:\university_tokyo\visual_media\segment-anything-main\outputs\seagrass-marine-biodiversity_overlay.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Original image not found: {image_path}")

if not os.path.isdir(mask_dir):
    raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

mask_files = sorted(
    [f for f in os.listdir(mask_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0])
)

print("Number of masks found:", len(mask_files))

if len(mask_files) == 0:
    raise RuntimeError("No mask png files found. Please check mask_dir.")

overlay = image.copy()

for filename in mask_files:
    mask_path = os.path.join(mask_dir, filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Failed to read:", filename)
        continue

    # 如果 mask 和原图尺寸不一致，自动调整
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    area = np.sum(mask > 0)

    # 太小的 mask 可以跳过，否则全是小噪点
    if area < 100:
        continue

    color = np.array([
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    ], dtype=np.uint8)

    alpha = 0.45

    # 只在 mask 区域上色
    overlay[mask > 0] = (
        image[mask > 0] * (1 - alpha) + color * alpha
    ).astype(np.uint8)

print("Saved overlay to:", output_path)
cv2.imwrite(output_path, overlay)