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


def divide_grid(warped, width, height):
    cell_width = width // 9
    cell_height = height // 9
    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = warped[i*cell_height:(i+1)*cell_height,
                          j*cell_width:(j+1)*cell_width]
            row.append(cell)
        cells.append(row)
    return cells


def apply_mask(frame, lower_hsv, upper_hsv):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(frame, frame, mask=mask)
    return result, mask


def adjust_mask_with_trackbar(warped):
    def nothing(x):
        pass

    cv.namedWindow("Trackbar")
    cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)

    while True:
        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")

        lower_hsv = np.array([l_h, l_s, l_v])
        upper_hsv = np.array([u_h, u_s, u_v])

        masked_frame, mask = apply_mask(warped, lower_hsv, upper_hsv)

        cv.imshow("Warped", warped)
        cv.imshow("Mask", mask)
        cv.imshow("Masked Frame", masked_frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


def main():
    frame = cv.imread("antrenare/1_01.jpg")
    width, height = 810, 810

    # Resize the frame for easier processing
    frame_resized = cv.resize(frame, (1024, 768))

    # Apply masking
    lower_hsv = np.array([0, 140, 0])
    upper_hsv = np.array([52, 255, 134])
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
        # Adjust mask on the warped image
        adjust_mask_with_trackbar(warped)
    else:
        print("Could not find a valid perspective transform.")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
