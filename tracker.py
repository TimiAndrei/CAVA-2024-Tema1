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

def process_frame(frame, reference_frame1, previous_frame, output_folder, frame_count):
    width, height = 2040, 2040

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
            if frame_count == 0:
                return further_warped
            
            # Detect differences
            if frame_count == 1 :
                # Compare with reference frames for the first image
                diff = cv.absdiff(further_warped, reference_frame1)
            else:
                # Compare with the previous frame for subsequent images
                diff = cv.absdiff(further_warped, previous_frame)

            # Convert to grayscale and threshold
            gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
            _, binary_diff = cv.threshold(gray_diff, 30, 255, cv.THRESH_BINARY)

            # save the differences
            diff_output_path = os.path.join(output_folder, f"diff_{frame_count:02d}.jpg")
            cv.imwrite(diff_output_path, binary_diff)

            # convert to binary_diff to bgr
            binary_diff_bgr = cv.cvtColor(binary_diff, cv.COLOR_GRAY2BGR)
            diff_output_path = os.path.join(output_folder, f"diff_{frame_count:02d}_color.jpg")
            cv.imwrite(diff_output_path, binary_diff_bgr)

            # save the differences with color
            binary_diff_color = cv.bitwise_and(further_warped, binary_diff_bgr)
            diff_output_path = os.path.join(output_folder, f"diff_{frame_count:02d}_color2.jpg")
            cv.imwrite(diff_output_path, binary_diff_color)

            # Extract the differences
            contours, _ = cv.findContours(binary_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05 and 80 <= w <= 150 and 80 <= h <= 150:
                    piece = further_warped[y:y+h, x:x+w]
                    piece_output_path = os.path.join(output_folder, f"piece_{frame_count:04d}_{i:02d}.jpg")
                    cv.imwrite(piece_output_path, piece)

            return further_warped
        else:
            print("Could not find a valid perspective transform for the warped image.")
            return None
    else:
        print("Could not find a valid perspective transform.")
        return None

def process_image(image_path, reference_frame1= None, previous_frame = None, output_folder = None, frame_count = 0):
    print(f"Processing {image_path}")
    frame = cv.imread(image_path)
    result = process_frame(frame, reference_frame1, previous_frame, output_folder, frame_count)
    if result is not None and output_folder is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv.imwrite(output_path, result)
    return result

def main():
    input_folder = "antrenare"
    output_folder = "warped_images"
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess and warp reference frames
    reference_frame_warped = process_image("imagini_auxiliare/01.jpg")

    if reference_frame_warped is None:
        print("Error processing reference frames.")
        return

    previous_frame = None
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    for frame_count, image_path in enumerate(image_paths):
        if frame_count == 0:
            continue
        adjusted_frame_count = (frame_count - 1) % 50 + 1
        previous_frame = process_image(image_path, reference_frame_warped, previous_frame, output_folder, adjusted_frame_count)

if __name__ == "__main__":
    main()