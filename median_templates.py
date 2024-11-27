import os
import cv2 as cv
import numpy as np
from collections import defaultdict


def load_classifications(folder):
    classifications = defaultdict(list)
    for filename in os.listdir(folder):
        if filename.endswith('.txt') and "scores" not in filename and "turns" not in filename:
            with open(os.path.join(folder, filename), 'r') as f:
                content = f.read().strip()
                position, classification = content.split()
                classifications[classification].append(
                    filename.replace('.txt', ''))
    return classifications


def load_images_by_classification(classifications, image_folder):
    images_by_classification = defaultdict(list)
    for classification, filenames in classifications.items():
        for filename in filenames:
            image_filename = f"piece_{filename}_cropped.jpg"
            image_path = os.path.join(image_folder, image_filename)
            if os.path.exists(image_path):
                image = cv.imread(image_path)
                if image is not None:
                    images_by_classification[classification].append(image)
                else:
                    print(f"Failed to load image: {image_path}")
            else:
                print(f"Image not found: {image_path}")
    return images_by_classification


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


def main():
    antrenare_folder = "antrenare"
    new_test2_folder = "new_try2"
    output_folder = "median_templates"

    # Load classifications from text files
    classifications = load_classifications(antrenare_folder)
    print(f"Loaded classifications: {classifications}")

    # Load images by classification
    images_by_classification = load_images_by_classification(
        classifications, new_test2_folder)
    print(
        f"Loaded images by classification: {images_by_classification.keys()}")

    # Generate and save median templates
    save_median_templates(images_by_classification, output_folder)


if __name__ == "__main__":
    main()
