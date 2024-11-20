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

def divide_grid(warped, width, height):
    cell_width = width // 9
    cell_height = height // 9
    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = warped[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            row.append(cell)
        cells.append(row)
    return cells

def apply_mask(frame):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 140, 0])
    upper_hsv = np.array([52, 255, 134])
    mask = cv.inRange(frame_hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(frame, frame, mask=mask)
    return result, mask

def main():
    frame = cv.imread("antrenare/1_01.jpg")
    width, height = 810, 810

    # Resize the frame for easier processing
    frame_resized = cv.resize(frame, (1024, 768))

    # Apply masking
    masked_frame, mask = apply_mask(frame_resized)

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

    warped = get_perspective_transform(frame_resized, max_contour, width, height)
    if warped is not None:
        # Display the warped image
        cv.imshow("Warped", warped)
        cv.waitKey(0)

        cells = divide_grid(warped, width, height)

        for row in cells:
            for cell in row:
                cv.imshow("Cell", cell)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                continue
            break
    else:
        print("Could not find a valid perspective transform.")

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()