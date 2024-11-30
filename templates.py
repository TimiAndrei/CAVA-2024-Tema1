import os
import cv2 as cv
import numpy as np
from utils import process_frame
from classifier import detect_bounding_box, get_centered_crop


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


def get_auxiliary_templates(all_pieces_by_classification):
    auxiliary_images = [
        {
            "path": "imagini_auxiliare/03.jpg",
            "positions": [
                "6E", "6F", "6G", "6H", "6I", "6J", "6K", "6L",
                "7E", "7F", "7G", "7H", "7I", "7J", "7K", "7L",
                "8E", "8F", "8G", "8H", "8I", "8J", "8K", "8L",
                "9E", "9F", "9G", "9H", "9I", "9J", "9K", "9L",
                "10E", "10F", "10G", "10H", "10I", "10J", "10K", "10L",
                "11E", "11F", "11G", "11H", "11I", "11J"
            ],
            "numbers": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90
            ]
        },
        {
            "path": "imagini_auxiliare/04.jpg",
            "positions": [
                "1A", "1C", "1E", "1G", "1I", "1K", "1M",
                "3A", "3C", "3E", "3G", "3I", "3K", "3M",
                "5A", "5C", "5E", "5G", "5I", "5K", "5M",
                "7A", "7C", "7E", "7G", "7I", "7K", "7M",
                "9A", "9C", "9E", "9G", "9I", "9K", "9M",
                "11A", "11C", "11E", "11G", "11I", "11K", "11M",
                "13A", "13C", "13E", "13G"
            ],
            "numbers": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90
            ]
        }
    ]

    output_folder = "templates3"
    os.makedirs(output_folder, exist_ok=True)

    cell_size = 145

    for aux_image in auxiliary_images:
        input_image_path = aux_image["path"]
        positions = aux_image["positions"]
        numbers = aux_image["numbers"]

        # Process the input image
        frame = cv.imread(input_image_path)
        warped_frame = process_frame(frame)

        if warped_frame is None:
            print("Error processing the auxiliary image:", input_image_path)
            continue

        for pos, num in zip(positions, numbers):
            row = int(pos[:-1]) - 1
            col = ord(pos[-1]) - ord('A')
            x_start = col * cell_size
            y_start = row * cell_size
            x_end = x_start + cell_size
            y_end = y_start + cell_size

            piece = warped_frame[y_start:y_end, x_start:x_end]

            bbox = detect_bounding_box(piece)
            if bbox:
                cropped_piece = get_centered_crop(piece, bbox, size=(120, 120))

                all_pieces_by_classification[num].append(cropped_piece)
            else:
                print(
                    f"Bounding box not found for piece {num} in {input_image_path}")
