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
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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

def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    zoomed_image = cv.resize(image, (new_width, new_height))

    # Crop the center of the zoomed image to maintain the original dimensions
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped_image = zoomed_image[start_y:start_y + height, start_x:start_x + width]
    return cropped_image

def process_frame(frame):
    width, height = 2030, 2030 # 14x14 grid of 145x145 cells

    # Resize the frame for easier processing
    frame_resized = cv.resize(frame, (1024, 768))

    # Apply masking
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([90, 255, 255])
    masked_frame, mask = apply_mask(frame_resized, lower_hsv, upper_hsv)

    # Preprocess the masked image
    blurred = preprocess_image(masked_frame)
    edges = detect_edges(blurred)

    max_contour = find_largest_contour(edges)

    warped = get_perspective_transform(frame_resized, max_contour, width, height)
    if warped is not None:
        # Zoom the warped image by 20%
        zoomed_warped = zoom_image(warped, 1.2)

        # Apply masking with specified HSV values on the zoomed warped image
        lower_hsv_warped = np.array([0, 0, 0])
        upper_hsv_warped = np.array([90, 130, 255])
        masked_warped, mask_warped = apply_mask(zoomed_warped, lower_hsv_warped, upper_hsv_warped)

        # Find squares in the mask of the zoomed warped image
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
        further_warped = get_perspective_transform(zoomed_warped, new_contour, width, height)
        if further_warped is not None:
            return further_warped
        else:
            print("Could not find a valid perspective transform for the warped image.")
            return None
    else:
        print("Could not find a valid perspective transform.")
        return None

def generate_templates():
    input_image_path = "imagini_auxiliare/04.jpg"
    output_folder = "templates"
    os.makedirs(output_folder, exist_ok=True)

    # Process the input image
    frame = cv.imread(input_image_path)
    warped_frame = process_frame(frame)
    #show warped_frame
    cv.imshow("Warped Frame", warped_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    if warped_frame is None:
        print("Error processing the input image.")
        return

    # Define the grid positions and corresponding numbers
    positions = [
        "1A", "1C", "1E", "1G", "1I", "1K", "1M",
        "3A", "3C", "3E", "3G", "3I", "3K", "3M",
        "5A", "5C", "5E", "5G", "5I", "5K", "5M",
        "7A", "7C", "7E", "7G", "7I", "7K", "7M",
        "9A", "9C", "9E", "9G", "9I", "9K", "9M",
        "11A", "11C", "11E", "11G", "11I", "11K", "11M",
        "13A", "13C", "13E", "13G"
    ]
    numbers = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90
    ]

    cell_size = 145
    for pos, num in zip(positions, numbers):
        row = int(pos[:-1]) - 1
        col = ord(pos[-1]) - ord('A')
        x_start = col * cell_size
        y_start = row * cell_size
        x_end = x_start + cell_size
        y_end = y_start + cell_size

        piece = warped_frame[y_start:y_end, x_start:x_end]
        piece_output_path = os.path.join(output_folder, f"{num}.jpg")
        cv.imwrite(piece_output_path, piece)

def main():
    generate_templates()

if __name__ == "__main__":
    main()