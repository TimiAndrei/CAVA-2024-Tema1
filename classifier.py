import cv2 as cv
import numpy as np
import os


def detect_bounding_box(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(max_contour)
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
        templates.append(template)
    return templates


def main():
    template_folder = "templates"
    templates = load_templates(template_folder)

    input_image_path = "./warped_images3/piece_1_02.jpg"
    image = cv.imread(input_image_path)

    best_match, best_score = process_and_classify(image, templates)
    if best_match is not None:
        print(f"Best match found with score: {best_score}")
        cv.imshow("Best Match", warp_to_standard_size(
            best_match, detect_bounding_box(best_match)))
        cv.imshow("Warped Image", warp_to_standard_size(
            image, detect_bounding_box(image)))
        cv.waitKey(0)
    else:
        print("No match found")


if __name__ == "__main__":
    main()
