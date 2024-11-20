import glob
import os
import cv2 as cv
import numpy as np


def preprocess_image(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_edges(blurred):
    edges = cv.Canny(blurred, 50, 150)
    return edges


def find_largest_contour(edges):
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    return max_contour


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_perspective_transform(frame, contour, width, height):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        ordered_pts = order_points(pts)
        pts1 = np.float32(ordered_pts)
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        warped = cv.warpPerspective(frame, M, (width, height))
        return warped
    else:
        return None


def apply_mask(frame, lower_hsv, upper_hsv):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(frame, frame, mask=mask)
    return result, mask


def find_squares(mask):
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv.contourArea(contour)
        # Aspect ratio close to 1 and area bigger than 50x50
        if 0.95 <= aspect_ratio <= 1.05 and 2500 <= area:
            squares.append((x, y, w, h))
    return squares


def find_corners(squares):
    top_left = min(squares, key=lambda s: s[0] + s[1])
    top_right = max(squares, key=lambda s: s[0] - s[1])
    bottom_left = min(squares, key=lambda s: s[0] - s[1])
    bottom_right = max(squares, key=lambda s: s[0] + s[1])
    return top_left, top_right, bottom_left, bottom_right


def process_frame(frame):
    width, height = 2040, 2040

    # Resize the frame for easier processing
    frame_resized = cv.resize(frame, (1024, 768))

    # Apply masking
    lower_hsv = np.array([0, 140, 0])
    upper_hsv = np.array([52, 255, 134])
    masked_frame, mask = apply_mask(frame_resized, lower_hsv, upper_hsv)

    # Preprocess the masked image
    blurred = preprocess_image(masked_frame)
    edges = detect_edges(blurred)

    max_contour = find_largest_contour(edges)

    warped = get_perspective_transform(
        frame_resized, max_contour, width, height)
    if warped is not None:
        # Apply masking with specified HSV values on the warped image
        lower_hsv_warped = np.array([0, 0, 0])
        upper_hsv_warped = np.array([90, 130, 255])
        masked_warped, mask_warped = apply_mask(
            warped, lower_hsv_warped, upper_hsv_warped)

        # Find squares in the mask of the warped image
        squares = find_squares(mask_warped)

        # Find corners of the largest square
        top_left, top_right, bottom_left, bottom_right = find_corners(squares)

        # Define the new contour for the further warped image
        new_contour = np.array([
            [top_left[0], top_left[1]],
            [top_right[0] + top_right[2], top_right[1]],
            [bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]],
            [bottom_left[0], bottom_left[1] + bottom_left[3]]
        ], dtype="float32")

        # Get perspective transform and warp the image again
        further_warped = get_perspective_transform(
            warped, new_contour, width, height)
        if further_warped is not None:
            return further_warped
        else:
            print("Could not find a valid perspective transform for the warped image.")
            return None
    else:
        print("Could not find a valid perspective transform.")
        return None


def process_image(image_path, output_folder):
    frame = cv.imread(image_path)
    result = process_frame(frame)
    if result is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv.imwrite(output_path, result)


def main():
    input_folder = "antrenare"
    output_folder = "warped_images"
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    for image_path in image_paths:
        process_image(image_path, output_folder)


if __name__ == "__main__":
    main()
