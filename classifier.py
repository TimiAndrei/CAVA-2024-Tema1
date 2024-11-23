import glob
import cv2 as cv
import numpy as np
import os


def detect_bounding_box(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)

    # Use morphological operations to remove noise and merge nearby components
    kernel = np.ones((5, 5), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # Ignore the margins (10%)
    h, w = morph.shape
    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)
    cropped_morph = morph[margin_h:h-margin_h, margin_w:w-margin_w]

    contours, _ = cv.findContours(
        cropped_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter out small contours that are likely to be noise or margins
        filtered_contours = [
            contour for contour in contours if cv.contourArea(contour) > 100]

        if filtered_contours:
            x_min, y_min = np.inf, np.inf
            x_max, y_max = -np.inf, -np.inf
            for contour in filtered_contours:
                x, y, w, h = cv.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            if x_min < np.inf and y_min < np.inf:
                # Adjust the bounding box coordinates to account for the cropped margins
                x, y, w, h = x_min + margin_w, y_min + margin_h, x_max - x_min, y_max - y_min
                return x, y, w, h
    return None


def warp_to_standard_size(image, bbox, size=(100, 100)):
    x, y, w, h = bbox
    rect = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype="float32")
    dst = np.array([
        [0, 0],
        [size[0] - 1, 0],
        [size[0] - 1, size[1] - 1],
        [0, size[1] - 1]
    ], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, size)
    return warped


def classify_number(template, image):
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)
    return max_val


def process_and_classify(image, templates, size=(100, 100)):
    bbox = detect_bounding_box(image)
    if bbox:
        warped_image = warp_to_standard_size(image, bbox, size)
        best_match = None
        best_score = -1
        for template in templates:
            warped_template = warp_to_standard_size(
                template, detect_bounding_box(template), size)
            score = classify_number(warped_template, warped_image)
            if score > best_score:
                best_score = score
                best_match = template
        return best_match, best_score
    return None, None


def load_templates(template_folder):
    templates = []
    for filename in os.listdir(template_folder):
        template_path = os.path.join(template_folder, filename)
        template = cv.imread(template_path)
        templates.append((template, filename))
    return templates


def save_warped_templates(input_folder, output_folder, size=(100, 100)):
    os.makedirs(output_folder, exist_ok=True)
    templates = load_templates(input_folder)
    for template, filename in templates:
        bbox = detect_bounding_box(template)
        if bbox:
            warped_template = warp_to_standard_size(template, bbox, size)
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, warped_template)


def save_warped_images(input_folder, output_folder, size=(100, 100)):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    for image_path in image_paths:
        image = cv.imread(image_path)
        bbox = detect_bounding_box(image)
        if bbox:
            warped_image = warp_to_standard_size(image, bbox, size)
            output_path = os.path.join(
                output_folder, os.path.basename(image_path))
            cv.imwrite(output_path, warped_image)


def main():
    # Save warped templates
    save_warped_templates("templates", "templates2")

    # Save warped images
    save_warped_images("warped_images3", "warped_images4")


if __name__ == "__main__":
    main()
