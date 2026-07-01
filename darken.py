from PIL import Image, ImageEnhance
import os

img = Image.open("assets/lexus.jpg").convert("RGB")

dark = ImageEnhance.Brightness(img).enhance(0.15)
dark.save("assets/lexus_dark.jpg")

low_contrast = ImageEnhance.Contrast(img).enhance(0.25)
low_contrast.save("assets/lexus_low_contrast.jpg")