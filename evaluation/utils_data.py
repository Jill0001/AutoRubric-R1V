import cv2
from PIL import Image, ImageDraw
import os


def draw_red_dot(image_path, out_path, x_relative, y_relative):
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist.")
        return
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    x = int(float(x_relative) * width)
    y = int(float(y_relative) * height)

    color = (0, 0, 255)
    radius = 10
    thickness = -1
    cv2.circle(image, (x, y), radius, color, thickness)

    cv2.imwrite(out_path, image)
    success = cv2.imwrite(out_path, image)
    if not success:
        print(f"Image {out_path} failed to save.")


def resize_image_to_fit(
    image_path, out_path, max_size=(1024, 512), resample_method=Image.LANCZOS
):
    image = Image.open(image_path)

    max_width, max_height = max_size

    original_width, original_height = image.size
    if original_width / original_height > max_width / max_height:
        ratio = max_width / original_width
    else:
        ratio = max_height / original_height

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), resample_method)

    resized_image.save(out_path)


def resize_and_crop(img, target_width, target_height):
    original_width, original_height = img.size

    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_height = target_height
        new_width = int(target_height * original_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = img.crop((0, 0, target_width, target_height))
    else:
        new_width = target_width
        new_height = int(target_width / original_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = img.crop((0, 0, target_width, target_height))
    return img
