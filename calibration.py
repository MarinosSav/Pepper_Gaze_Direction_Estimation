import cv2
import numpy as np
import pickle
from pepper_connector import socket_connection

"""Set-Up Variables"""
horizontal_corners = 4
vertical_corners = 3
chessboard_width = 7.5  # cm
isPepper = True

"""Global Variables"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_points_3D = np.zeros((horizontal_corners * vertical_corners, 3), np.float32)
object_points_3D[:, :2] = np.mgrid[0:vertical_corners, 0:horizontal_corners].T.reshape(-1, 2)
camera_calibration = {}
object_points = []
image_points = []
images = []
if isPepper:
    connect = socket_connection(ip='192.168.0.180', port=12345, camera=4)
else:
    connect = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loop through video frames
while True:
    if isPepper:
        frame = connect.get_img()
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
    else:
        success, frame = connect.read()
        if not success:
            break

    cv2.imshow("", frame)

    key = cv2.waitKey(30)
    # When the space bar is pressed take a screenshot and store it
    if key == 32:
        images.append(frame)
    # When esc is pressed exit
    if key == 27:
        # If no pictures where taken, do not exit
        if len(images) == 0:
            print("No pictures stored")
        else:
            break

frame_height, frame_width, _ = frame.shape
# For each image stored find the chessboard corners
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (vertical_corners, horizontal_corners), None)
    if found:
        object_points.append(object_points_3D)
        better_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)
        cv2.drawChessboardCorners(image, (vertical_corners, horizontal_corners), better_corners, found)

_, camera_matrix, dist_coeffs, rvec, tvec = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None,
                                                                None)

image_step = 0
found = False

def step():
    """Helper function used to traverse screenshots stored. It returns the next screenshot."""

    global image_step

    if image_step == len(images) - 1:
        image_step = 0
    else:
        image_step += 1

    return

while True:
    key = cv2.waitKey(30)

    # Traverse coordinates of chessboard corner points
    for points in image_points:
        sorted_x = np.sort(points[:, :, 0])
        # Find a screenshot where the chessboard orientation is as straight as possible
        if abs(sorted_x[0] - sorted_x[1]) < 1 and abs(sorted_x[0] - sorted_x[2]) < 1 and not found:
            # Find ratio relationship between real world measurements and digital pixel representation
            found = True
            width = (abs(points[9][0][0] - points[0][0][0]) + abs(points[10][0][0] - points[1][0][0]) +
                     abs(points[11][0][0] - points[2][0][0])) / 3
            real_over_digital = chessboard_width / width
            print("Ratio: ", real_over_digital)
            cv2.line(images[image_step], (int(points[0][0][0]), int(points[0][0][1])),
                     (int(points[0][0][0] + width), int(points[0][0][1])), (255, 0, 0), 3)
            cv2.imshow("Found", images[image_step])
        step()

    cv2.imshow("", images[image_step])
    if key == 32:
        step()
    if key == 27:
        break

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficient:\n", dist_coeffs)

# Store the camera matrix, distance coefficients and real over digital ratio for future use
camera_calibration["camera_matrix"] = camera_matrix
camera_calibration["dist_coeffs"] = dist_coeffs
if found:
    camera_calibration["real_over_digital"] = real_over_digital
pickle.dump(camera_calibration, open("camera_calibration.p", "wb"))
if isPepper:
    connect.close_connection()
cv2.destroyAllWindows()
