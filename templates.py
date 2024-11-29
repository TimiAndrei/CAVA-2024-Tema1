import os
import cv2 as cv
import numpy as np
from collections import defaultdict


def generate_median_template(images):
    if not images:
        return None
    stack = np.stack(images, axis=3)
    median_image = np.median(stack, axis=3).astype(np.uint8)
    return median_image


def save_median_templates(images_by_classification, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for classification, images in images_by_classification.items():
        print(
            f"Generating median template for classification: {classification} with {len(images)} images")
        median_template = generate_median_template(images)
        if median_template is not None:
            output_path = os.path.join(output_folder, f"{classification}.jpg")
            cv.imwrite(output_path, median_template)
            print(f"Saved median template to: {output_path}")
        else:
            print(
                f"No images to generate median template for classification: {classification}")


# def generate_templates():
#     input_image_path = "imagini_auxiliare/03.jpg"
#     output_folder = "templates3"
#     os.makedirs(output_folder, exist_ok=True)

#     # Process the input image
#     frame = cv.imread(input_image_path)
#     warped_frame = process_frame(frame)
#     cv.imwrite("full_board.jpg", warped_frame)

#     if warped_frame is None:
#         print("Error processing the input image.")
#         return

#     # Define the grid positions and corresponding numbers
#     positions = [
#         "6E", "6F", "6G", "6H", "6I", "6J", "6K", "6L",
#         "7E", "7F", "7G", "7H", "7I", "7J", "7K", "7L",
#         "8E", "8F", "8G", "8H", "8I", "8J", "8K", "8L",
#         "9E", "9F", "9G", "9H", "9I", "9J", "9K", "9L",
#         "10E", "10F", "10G", "10H", "10I", "10J", "10K", "10L",
#         "11E", "11F", "11G", "11H", "11I", "11J"
#     ]
#     numbers = [
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90
#     ]

#     cell_size = 145
#     for pos, num in zip(positions, numbers):
#         row = int(pos[:-1]) - 1
#         col = ord(pos[-1]) - ord('A')
#         x_start = col * cell_size
#         y_start = row * cell_size
#         x_end = x_start + cell_size
#         y_end = y_start + cell_size

#         piece = warped_frame[y_start:y_end, x_start:x_end]

#         # Detect bounding box and get centered crop
#         bbox = detect_bounding_box(piece)
#         if bbox:
#             cropped_piece = get_centered_crop(piece, bbox, size=(120, 120))
#             piece_output_path = os.path.join(output_folder, f"{num}.jpg")
#             cv.imwrite(piece_output_path, cropped_piece)
#         else:
#             print(f"Bounding box not found for piece {num}")
