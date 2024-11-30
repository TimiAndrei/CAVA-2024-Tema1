from collections import defaultdict
import glob
import os
import cv2 as cv
import numpy as np


def detect_edges(blurred):
    edges = cv.Canny(blurred, 50, 150)
    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    thick_edges = cv.dilate(edges, dilation_kernel, iterations=4)

    return thick_edges


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
    height = mask.shape[0]
    top_20_percent = height * 0.2
    bottom_20_percent = height * 0.8

    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv.contourArea(contour)
        # At least squares, but can be rectangles
        if 0.5 <= aspect_ratio <= 2.00 and 2500 <= area:
            # Check if the square is in the top 20% or bottom 20% of the image
            if y <= top_20_percent or y >= bottom_20_percent:
                squares.append((x, y, w, h))
    return squares


def find_corners(squares):
    top_left = min(squares, key=lambda s: s[0] + s[1])
    top_right = max(squares, key=lambda s: s[0] - s[1])
    bottom_left = min(squares, key=lambda s: s[0] - s[1])
    bottom_right = max(squares, key=lambda s: s[0] + s[1])
    return top_left, top_right, bottom_left, bottom_right


def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    zoomed_image = cv.resize(image, (new_width, new_height))

    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped_image = zoomed_image[start_y:start_y +
                                 height, start_x:start_x + width]
    return cropped_image


def process_frame(frame):
    width, height = 2030, 2030  # 14x14 grid of 145x145 cells

    # Apply masking
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([70, 255, 255])
    masked_frame, mask = apply_mask(frame, lower_hsv, upper_hsv)

    # Preprocess the masked image
    edges = detect_edges(masked_frame)

    max_contour = find_largest_contour(edges)
    if max_contour is None:
        print("No contours found.")
        return None

    # # Draw the largest contour on the original image for debugging
    # contour_image = frame.copy()
    # cv.drawContours(contour_image, [max_contour], -1, (0, 255, 0), 30)
    # contour_image = cv.resize(contour_image, (640, 480))
    # cv.imshow("Contour", contour_image)
    # cv.waitKey(0)

    warped = get_perspective_transform(frame, max_contour, width, height)
    if warped is not None:
        # # Display the warped image for debugging
        # resized_warped = cv.resize(warped, (640, 480))
        # cv.imshow("Warped", resized_warped)
        # cv.waitKey(0)

        # Zoom the warped image by 20%
        zoomed_warped = zoom_image(warped, 1.3)

        # Apply masking with specified HSV values on the zoomed warped image
        lower_hsv_warped = np.array([0, 0, 0])
        upper_hsv_warped = np.array([70, 130, 255])
        masked_warped, mask_warped = apply_mask(
            zoomed_warped, lower_hsv_warped, upper_hsv_warped)

        # Find squares in the mask of the zoomed warped image
        squares = find_squares(mask_warped)

        if squares:
            # Find corners of the largest square
            top_left, top_right, bottom_left, bottom_right = find_corners(
                squares)

            # Define the new contour for the further warped image
            new_contour = np.array([
                [top_left[0], top_left[1]],
                [top_right[0] + top_right[2], top_right[1]],
                [bottom_right[0] + bottom_right[2],
                    bottom_right[1] + bottom_right[3]],
                [bottom_left[0], bottom_left[1] + bottom_left[3]]
            ], dtype="float32")

            # Get perspective transform and warp the image again
            further_warped = get_perspective_transform(
                zoomed_warped, new_contour, width, height)

            if further_warped is not None:
                return further_warped
            else:
                print(
                    "Could not find a valid perspective transform for the warped image.")
                return None
        else:
            print("No squares found in the mask of the zoomed warped image.")
            return None
    else:
        print("Could not find a valid perspective transform.")
        return None
