from collections import defaultdict
import glob
import os
import cv2 as cv
import numpy as np
from classifier import detect_bounding_box, get_centered_crop, load_templates, process_and_classify
from templates import save_median_templates
from utils import apply_mask, detect_edges, find_largest_contour, get_perspective_transform, find_squares, find_corners, zoom_image


def process_frame(frame):
    width, height = 2030, 2030  # 14x14 grid of 145x145 cells

    # Apply masking
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([95, 255, 255])
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
        upper_hsv_warped = np.array([90, 130, 255])
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


def compare_and_extract_pieces(current_frame, previous_frame, output_folder, image_name, templates):
    cell_size = 145
    grid_size = 14
    max_diff = 0
    max_diff_cell = None

    for i in range(grid_size):
        for j in range(grid_size):
            x_start = j * cell_size
            y_start = i * cell_size
            x_end = x_start + cell_size
            y_end = y_start + cell_size

            current_cell = current_frame[y_start:y_end, x_start:x_end]
            previous_cell = previous_frame[y_start:y_end, x_start:x_end]

            diff = cv.absdiff(current_cell, previous_cell)
            diff_sum = np.sum(diff)

            if diff_sum > max_diff:
                max_diff = diff_sum
                max_diff_cell = (x_start, y_start, x_end, y_end, i, j)

    if max_diff_cell:
        x_start, y_start, x_end, y_end, row, col = max_diff_cell
        piece = current_frame[y_start:y_end, x_start:x_end]
        piece_output_path = os.path.join(
            output_folder, f"piece_{image_name}.jpg")
        cv.imwrite(piece_output_path, piece)
        cropped_piece = get_centered_crop(
            piece, detect_bounding_box(piece), size=(120, 120))
        cropped_piece_output_path = os.path.join(
            output_folder, f"piece_{image_name}_cropped.jpg")
        cv.imwrite(cropped_piece_output_path, cropped_piece)

        # Determine the grid position
        col_letter = chr(ord('A') + col)
        row_number = row + 1
        position = f"{row_number}{col_letter}"

        # Classify the piece
        matches_and_scores = process_and_classify(piece, templates)
        if matches_and_scores:
            best_match, best_score = matches_and_scores[0]
            best_match_filename = os.path.splitext(best_match)[0]
        else:
            best_match_filename = "unknown"

        # Optionally, you can print or log all matches and scores
        for match, score in matches_and_scores:
            print(f"Template: {match}, Score: {score}")
        # Write the position and classification to a text file
        text_output_path = os.path.join(
            output_folder, f"piece_{image_name}.txt")
        with open(text_output_path, 'w') as f:
            f.write(f"{position} {best_match_filename}")


def process_image(image_path, previous_frame, output_folder, templates):
    print(f"Processing {image_path}")
    frame = cv.imread(image_path)
    warped_frame = process_frame(frame)
    if warped_frame is not None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if previous_frame is not None:
            compare_and_extract_pieces(
                warped_frame, previous_frame, output_folder, image_name, templates)
    return warped_frame


def get_pieces(image_path, previous_frame):
    cell_size = 145
    grid_size = 14
    pieces_by_classification = defaultdict(list)

    image_name = os.path.basename(image_path)
    txt_path = image_path.replace('.jpg', '.txt')
    current_frame = cv.imread(image_path)
    current_frame = process_frame(current_frame)

    if current_frame is None:
        print(f"Failed to load image: {image_path}")
        return pieces_by_classification, previous_frame

    if not os.path.exists(txt_path):
        print(f"Classification file not found: {txt_path}")
        return pieces_by_classification, previous_frame

    with open(txt_path, 'r') as f:
        content = f.read().strip()
        position, classification = content.split()
        classification = int(classification)

    max_diff = 0
    max_diff_cell = None

    for i in range(grid_size):
        for j in range(grid_size):
            x_start = j * cell_size
            y_start = i * cell_size
            x_end = x_start + cell_size
            y_end = y_start + cell_size

            current_cell = current_frame[y_start:y_end, x_start:x_end]
            if previous_frame is not None:
                previous_cell = previous_frame[y_start:y_end, x_start:x_end]
                diff = cv.absdiff(current_cell, previous_cell)
                diff_sum = np.sum(diff)

                if diff_sum > max_diff:
                    max_diff = diff_sum
                    max_diff_cell = (x_start, y_start, x_end, y_end, i, j)

    if max_diff_cell:
        x_start, y_start, x_end, y_end, row, col = max_diff_cell
        piece = current_frame[y_start:y_end, x_start:x_end]

        bbox = detect_bounding_box(piece)
        if bbox:
            cropped_piece = get_centered_crop(piece, bbox, size=(120, 120))
            # Add the piece to the dictionary

            pieces_by_classification[classification].append(cropped_piece)
            print(f"Classified piece {image_name} as {classification}")
        else:
            print(f"Bounding box not found for piece {image_name}")

    return pieces_by_classification, current_frame


def generate_templates():
    input_folder = "antrenare"
    output_folder = "new_median_templates"
    os.makedirs(output_folder, exist_ok=True)

    # Load and process the empty board
    empty_board = cv.imread("imagini_auxiliare/01.jpg")
    empty_board_warped = process_frame(empty_board)
    cv.imwrite("empty_board_warped.jpg", empty_board_warped)

    previous_frame = empty_board_warped
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    all_pieces_by_classification = defaultdict(list)

    for frame_count, image_path in enumerate(image_paths):
        if frame_count % 50 == 0:
            # Reset the base frame every 50 images
            previous_frame = empty_board_warped

        # Load the image
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            continue

        pieces_by_classification, previous_frame = get_pieces(
            image_path, previous_frame)

        # Merge the pieces into the main dictionary
        for classification, pieces in pieces_by_classification.items():
            all_pieces_by_classification[classification].extend(pieces)

    save_median_templates(all_pieces_by_classification, output_folder)


def generate_output():
    input_folder = "antrenare"
    output_folder = "new_try3"
    os.makedirs(output_folder, exist_ok=True)

    # Load and process the empty board
    empty_board = cv.imread("imagini_auxiliare/01.jpg")
    empty_board_warped = process_frame(empty_board)
    cv.imwrite("empty_board_warped.jpg", empty_board_warped)

    previous_frame = empty_board_warped
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    templates = load_templates("new_median_templates")
    for frame_count, image_path in enumerate(image_paths):
        if frame_count % 50 == 0:
            # Reset the base frame every 50 images
            previous_frame = empty_board_warped
        previous_frame = process_image(
            image_path, previous_frame, output_folder, templates)


def main():
    generate_templates()
    generate_output()


if __name__ == "__main__":
    main()
