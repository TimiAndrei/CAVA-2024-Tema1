import glob
import os
import cv2 as cv
import numpy as np
from collections import defaultdict
from classifier import detect_bounding_box, get_centered_crop, load_templates, process_and_classify
from templates import generate_templates
from utils import process_frame

# Special tiles and their positions
special_tiles = {
    "3x": ["1A", "1G", "1H", "1N", "7A", "7N", "8A", "8N", "14A", "14G", "14H", "14N"],
    "2x": ["2B", "2M", "3C", "3L", "4D", "4K", "5E", "5J", "10E", "10J", "11D", "11K", "12C", "12L", "13B", "13M"],
}

board_size = 14

# Initialize the board with starting numbers
initial_board = np.full((board_size, board_size), "#", dtype=object)
initial_board[6, 6] = 1  # 7G
initial_board[6, 7] = 2  # 7H
initial_board[7, 6] = 3  # 8G
initial_board[7, 7] = 4  # 8H


def parse_turns(file_path):
    turns = []
    with open(file_path, 'r') as f:
        for line in f:
            player, turn = line.strip().split()
            turns.append((player, int(turn)))
    return turns


def get_position_indices(position):
    row = int(position[:-1]) - 1
    col = ord(position[-1]) - ord('A')
    return row, col


def calculate_score(piece, position, board):
    row, col = get_position_indices(position)
    score = 0

    # Check for special tiles
    position_str = f"{row + 1}{chr(col + ord('A'))}"
    multiplier = 1
    if position_str in special_tiles["3x"]:
        multiplier = 3
    elif position_str in special_tiles["2x"]:
        multiplier = 2

    # Helper function to check equations
    def check_equation(r1, c1, r2, c2):
        nonlocal score
        if 0 <= r1 < board_size and 0 <= c1 < board_size and 0 <= r2 < board_size and 0 <= c2 < board_size:
            if board[r1, c1] == "#" or board[r2, c2] == "#":
                return  # Ignore empty positions
            if board[r1, c1] + board[r2, c2] == piece:
                print(
                    f"Found equation: {board[r1, c1]} + {board[r2, c2]} == {piece}")
                score += piece
                print("Score:", score)
            elif abs(board[r1, c1] - board[r2, c2]) == piece:
                print(
                    f"Found equation: |{board[r1, c1]} - {board[r2, c2]}| == {piece}")
                score += piece
                print("Score:", score)
            elif board[r2, c2] != 0 and board[r1, c1] // board[r2, c2] == piece:
                print(
                    f"Found equation: {board[r1, c1]} // {board[r2, c2]} == {piece}")
                score += piece
                print("Score:", score)
            elif board[r1, c1] != 0 and board[r2, c2] // board[r1, c1] == piece:
                print(
                    f"Found equation: {board[r2, c2]} // {board[r1, c1]} == {piece}")
                score += piece
                print("Score:", score)
            elif board[r1, c1] * board[r2, c2] == piece:
                print(
                    f"Found equation: {board[r1, c1]} * {board[r2, c2]} == {piece}")
                score += piece
                print("Score:", score)

    # Check for valid equations
    # Vertical
    print(f"Checking up:...")
    check_equation(row - 2, col, row - 1, col)  # Up
    print(f"Checking down:...")
    check_equation(row + 2, col, row + 1, col)  # Down

    # Horizontal
    print(f"Checking left:...")
    check_equation(row, col - 2, row, col - 1)  # Left
    print(f"Checking right:...")
    check_equation(row, col + 2, row, col + 1)  # Right

    # Apply the multiplier for special tiles
    score *= multiplier
    if multiplier > 1:
        print(f"Multiplier applied: {multiplier}x at {position_str}")

    print("Final score:", score)
    print(board)
    return score


def compare_and_extract_pieces(current_frame, previous_frame, output_folder, image_name, templates, board):
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
            best_match, _ = matches_and_scores[0]
            best_match_filename = os.path.splitext(best_match)[0]
            piece_value = int(best_match_filename)
        else:
            best_match_filename = "unknown"
            piece_value = 0

        # Calculate the score for the piece
        score = calculate_score(piece_value, position, board)

        # Update the board with the new piece
        board[row, col] = piece_value

        # Write the position and classification to a text file
        text_output_path = os.path.join(output_folder, f"{image_name}.txt")
        with open(text_output_path, 'w') as f:
            f.write(f"{position} {best_match_filename}")

        return score
    return 0


def process_image(image_path, previous_frame, output_folder, templates, board):
    print(f"Processing {image_path}")
    frame = cv.imread(image_path)
    warped_frame = process_frame(frame)
    if warped_frame is not None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if previous_frame is not None:
            score = compare_and_extract_pieces(
                warped_frame, previous_frame, output_folder, image_name, templates, board)
            return warped_frame, score
    return warped_frame, 0


def generate_output():
    input_folder = "antrenare"
    output_folder = "evaluare/fisiere_solutie/464_Andrei_Timotei/"
    os.makedirs(output_folder, exist_ok=True)

    # Load and process the empty board
    empty_board = cv.imread("imagini_auxiliare/01.jpg")
    empty_board_warped = process_frame(empty_board)

    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    templates = load_templates("new_median_templates")

    game_number = 1
    current_turn = 1
    current_player = None
    cumulative_score = 0
    starting_turn = 1
    previous_frame = empty_board_warped
    board = initial_board.copy()
    first_round_processed = False

    turns_file_path = os.path.join(input_folder, f"{game_number}_turns.txt")
    turns = parse_turns(turns_file_path)

    for frame_count, image_path in enumerate(image_paths):
        if frame_count % 50 == 0 and frame_count > 0:
            # Write the scores to the output file for the previous game
            with open(os.path.join(output_folder, f"{game_number}_scores.txt"), 'a') as f:
                if current_player is not None and first_round_processed:
                    f.write(
                        f"{current_player} {starting_turn} {cumulative_score}\n")
            game_number += 1
            current_turn = 1
            current_player = None
            cumulative_score = 0
            starting_turn = 1
            first_round_processed = False

            # Reset the base frame and board for the new game
            previous_frame = empty_board_warped
            board = initial_board.copy()

            # Parse the turns file for the new game
            turns_file_path = os.path.join(
                input_folder, f"{game_number}_turns.txt")
            if not os.path.exists(turns_file_path):
                break
            turns = parse_turns(turns_file_path)
            print(turns)

            # Initialize the first player and starting turn
            current_player, starting_turn = turns[0]

        previous_frame, score = process_image(
            image_path, previous_frame, output_folder, templates, board)

        # Determine the current player based on the turns list
        for player, turn in turns:
            if current_turn == turn:
                if current_player is not None and first_round_processed:
                    # Write the cumulative score for the previous player
                    with open(os.path.join(output_folder, f"{game_number}_scores.txt"), 'a') as f:
                        f.write(
                            f"{current_player} {starting_turn} {cumulative_score}\n")
                # Reset cumulative score and update current player and starting turn
                cumulative_score = 0
                current_player = player
                starting_turn = turn
                first_round_processed = True

        # Update the score for the current player
        cumulative_score += score
        print(
            f"Player: {current_player}, Turn: {current_turn}, Score: {cumulative_score}")

        current_turn += 1

    # Write the scores for the last game
    with open(os.path.join(output_folder, f"{game_number}_scores.txt"), 'a') as f:
        if current_player is not None and first_round_processed:
            f.write(f"{current_player} {starting_turn} {cumulative_score}\n")


def main():
    # generate_templates()
    generate_output()


if __name__ == "__main__":
    main()
