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
        # Aspect ratio close to 1 and area bigger than 50x50
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

    # Crop the center of the zoomed image to maintain the original dimensions
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped_image = zoomed_image[start_y:start_y +
                                 height, start_x:start_x + width]
    return cropped_image


def process_frame(frame):
    width, height = 2040, 2040

    # Resize the frame for easier processing
    frame_resized = cv.resize(frame, (1024, 768))

    # Apply masking
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([90, 255, 255])
    masked_frame, mask = apply_mask(frame_resized, lower_hsv, upper_hsv)

    # Display the masked image
    cv.imshow("Masked", masked_frame)
    cv.waitKey(0)

    # Preprocess the masked image
    blurred = preprocess_image(masked_frame)
    edges = detect_edges(blurred)

    # Display the edges
    cv.imshow("Edges", edges)
    cv.waitKey(0)

    max_contour = find_largest_contour(edges)

    # Draw the largest contour on the original image
    contour_image = frame_resized.copy()
    cv.drawContours(contour_image, [max_contour], -1, (0, 255, 0), 3)
    cv.imshow("Contour", contour_image)
    cv.waitKey(0)

    warped = get_perspective_transform(
        frame_resized, max_contour, width, height)
    if warped is not None:
        # Display the warped image
        resized_warped = cv.resize(warped, (640, 480))
        cv.imshow("Warped", resized_warped)
        cv.waitKey(0)

        zoomed_warped = zoom_image(warped, 1.2)
        cv.imshow("Zoomed Warped", zoomed_warped)
        cv.waitKey(0)

        # Apply masking with specified HSV values on the zoomed warped image
        lower_hsv_warped = np.array([0, 0, 0])
        upper_hsv_warped = np.array([90, 130, 255])
        masked_warped, mask_warped = apply_mask(
            zoomed_warped, lower_hsv_warped, upper_hsv_warped)

        # Display the masked zoomed warped image
        resized_masked_warped = cv.resize(masked_warped, (640, 480))
        cv.imshow("Masked Zoomed Warped", resized_masked_warped)
        cv.waitKey(0)

        # Find squares in the mask of the zoomed warped image
        squares = find_squares(mask_warped)

        # Draw squares on the zoomed warped image
        for (x, y, w, h) in squares:
            cv.rectangle(zoomed_warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        resized_warped_with_squares = cv.resize(zoomed_warped, (640, 480))
        cv.imshow("Squares Zoomed Warped", resized_warped_with_squares)
        cv.waitKey(0)

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
            zoomed_warped, new_contour, width, height)
        if further_warped is not None:
            # Display the further warped image
            resized_further_warped = cv.resize(further_warped, (640, 480))
            cv.imshow("Further Warped", resized_further_warped)
            cv.waitKey(0)
        else:
            print("Could not find a valid perspective transform for the warped image.")
    else:
        print("Could not find a valid perspective transform.")

    cv.destroyAllWindows()


def main():
    # frame = cv.imread("antrenare/3_34.jpg")
    frame = cv.imread("imagini_auxiliare/04.jpg")
    process_frame(frame)


if __name__ == "__main__":
    main()
