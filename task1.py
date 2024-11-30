from collections import defaultdict
import glob
import os
import cv2 as cv
import numpy as np
from classifier import detect_bounding_box, get_centered_crop, load_templates, process_and_classify
from templates import generate_templates
from utils import process_frame


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

        cropped_piece = get_centered_crop(
            piece, detect_bounding_box(piece), size=(120, 120))

        # Determine the grid position
        col_letter = chr(ord('A') + col)
        row_number = row + 1
        position = f"{row_number}{col_letter}"

        # Classify the piece
        matches_and_scores = process_and_classify(cropped_piece, templates)
        if matches_and_scores:
            best_match, best_score = matches_and_scores[0]
            best_match_filename = os.path.splitext(best_match)[0]
        else:
            best_match_filename = "unknown"

        # Log all matches and scores
        # for match, score in matches_and_scores:
        #     print(f"Template: {match}, Score: {score}")
        # Write the position and classification to a text file
        text_output_path = os.path.join(
            output_folder, f"{image_name}.txt")
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


def generate_output():
    input_folder = "evaluare/fake_test"
    output_folder = "evaluare/fisiere_solutie/464_Andrei_Timotei/"
    os.makedirs(output_folder, exist_ok=True)

    # Load and process the empty board
    empty_board = cv.imread("imagini_auxiliare/01.jpg")
    empty_board_warped = process_frame(empty_board)

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
