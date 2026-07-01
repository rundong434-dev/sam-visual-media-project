from PIL import Image, ImageEnhance
import os

input_path = "assets/lexus_dark.jpg"
output_path = "assets/lexus_dark_enhanced.jpg"

img = Image.open(input_path).convert("RGB")

img = ImageEnhance.Brightness(img).enhance(2.0)
img = ImageEnhance.Contrast(img).enhance(1.6)
img = ImageEnhance.Sharpness(img).enhance(1.3)

img.save(output_path)