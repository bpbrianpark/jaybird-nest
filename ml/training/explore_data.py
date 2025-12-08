from PIL import Image
import os

image_path = "path/to/falcons/image1.jpg"
img = Image.open(image_path)
print(f"Size: {img.size}")
print(f"Format: {img.format}")
print(f"Mode: {img.mode}")
img.show()

