import cv2
import dlib
import numpy as np
from pepper_connector import socket_connection
import math
import time
import pickle
import pygame
import random
import pandas as pd

"""Set-up Variables"""
isPepper = False  # False -> laptop camera settings will be used | True -> pepper camera settings will be used
testMode = False  # Use True when running experiments
debug_mode = "basic"  # Available settings (right-most being most fps limiting): "none", "basic", "angles", "full"
real_distance = 50  # Real-world distance in cm

"""Test Variables"""
test_number = 12
test_name = ""

"""Constants"""
if isPepper:
    connect = socket_connection(ip='192.168.0.180', port=12345, camera=4)
    if real_distance == 120:
        real_over_digital = pickle.load(open("pep_120.p", "rb"))["real_over_digital"]
    else:
        real_over_digital = pickle.load(open("pep_80.p", "rb"))["real_over_digital"]
    focal_length = pickle.load(open("pepper.p", "rb"))["camera_matrix"][0][0]
else:
    connect = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if real_distance == 50:
        real_over_digital = pickle.load(open("pc_50.p", "rb"))["real_over_digital"]
    elif real_distance == 70:
        real_over_digital = pickle.load(open("pc_70.p", "rb"))["real_over_digital"]
    else:
        real_over_digital = pickle.load(open("pc_40.p", "rb"))["real_over_digital"]
    focal_length = pickle.load(open("pc.p", "rb"))["camera_matrix"][0][0]
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
if testMode:
    # Initialize pyGame
    pygame.init()
    start_time = pygame.time.get_ticks()
    screen_info = pygame.display.Info()

"""Global Variables"""
drawPoint = False
testStopped = False
testPaused = False
test_point = []
test_stage = 0
start_error_timer = 0
paused_time = 0
points_left = []
frame_num = 0
test_results = []
store_yaw = [0, []]
store_pitch = [0, []]
store_eye_angle_x = [0, []]
store_eye_angle_y = [0, []]
previous_x_result = 1
previous_x_error = 10
previous_y_result = 1
previous_y_error = 10
previous_eye_x = 1
previous_eye_x_error = 0.1
previous_eye_y = 1
previous_eye_y_error = 0.1


def show(name, img, scale=5):
    """Just a helper function. It resizes an image (img) based on the scale factor (scale) and creates a viewable
    window using the name (name) provided. In the case a scale factor is not passed in x5 is used."""

    if (debug_mode == "full" or (name == "frame" and debug_mode == "basic")) and not testMode:
        try:
            cv2.imshow(str(name), cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))))
        except:
            pass

    return


def to_pixel(metric, unit="cm"):
    """Converts real-world distances to digital distances so they can be used in equations. It takes as input the
    real-world distance (metric) and the real-world measurement unit (unit) and returns a digital pixel-based distance.
    Available conversions include cm, mm and m. If no unit measurement is passed in, cm is used by default. Camera
    calibration is needed to produce accurate results."""

    result = metric / real_over_digital
    if unit == 'm':
        return result * 100
    elif unit == 'mm':
        return result / 10

    return result


def kalman(measurement, previous_estimate, previous_error, error_in_measurement):
    """Applies Kalman filtering to a measurement, reducing noise significantly with repeated observations. It takes as
    input the measurement to be corrected (measurement), the previous estimated made by this function
    (previous_estimate), the previous error in estimation made by this function (previous_error) and a fixed amount of
    error in measurement (error_in_measurement) and returns a better prediction for the given measurement using previous
    data as well as the new error in calculation to be used in future executions."""

    kalman_gain = previous_estimate / (previous_estimate + error_in_measurement)
    new_estimate = previous_estimate + kalman_gain * (measurement - previous_estimate)

    new_error = (1 - kalman_gain) * previous_error

    return new_estimate, new_error


def stabilize_angle(angle, previous_angles, conf_interval, error_margin):
    """Reduces angle variation that may be caused by environmental jitter. It takes as input the angle that we would
    like to stabilize (angle), a list of previous readings for that angle (previous_angles), a confidence interval
    (conf_interval) and an error margin (error_margin), and returns the stabilized angle. The confidence interval is
    the interval at which we can confidently say that two angles are similar and the error margin is the margin at which
    an angle given is too varied and henceforth a potential error."""

    queue_max_length = 10

    if not previous_angles[1]:
        previous_angles[1].append(angle)
        previous_angles[0] = 0
        return angle
    angle_mean = abs(np.mean(previous_angles[1]))
    # If angle given is outside the error margin ignore it and return the previous stored value
    if abs(abs(angle) - angle_mean) > error_margin:
        if type(previous_angles[1]) == list:
            return previous_angles[1][-1]
        return previous_angles[1]
    # If angle given is outside the confidence interval then refresh the queue (new point)
    if abs(abs(angle) - angle_mean) > conf_interval:
        previous_angles[1].clear()
        previous_angles[1].append(angle)
        previous_angles[0] = 0
        return angle
    # If the queue length has been exceeded then replace the oldest value
    if len(previous_angles[1]) == queue_max_length:
        previous_angles[1][previous_angles[0]] = angle
        if previous_angles[0] == queue_max_length - 1:
            previous_angles[0] = 0
        else:
            previous_angles[0] += 1
    else:
        # If angle is within the error margin then add it to the queue
        previous_angles[1].append(angle)

    return np.mean(previous_angles[1])


def run_test():
    """Creates testing infrastructure for testing the code and monitors the tests. It returns the coordinates of the
    point that is being tested."""

    global testStopped, testPaused, drawPoint, test_point, points_left, paused_time, test_stage, start_error_timer,\
        screen_info
    # List of points that will be showed on screen
    potential_points = [(((screen_info.current_w / 2) - (frame_width / 2)), screen_info.current_h / 2),
                        (((screen_info.current_w / 2) + (frame_width / 2)), screen_info.current_h / 2),
                        (screen_info.current_w / 2, screen_info.current_h / 2),
                        (screen_info.current_w / 2, ((screen_info.current_h / 2) - (frame_height / 2))),
                        (screen_info.current_w / 2, ((screen_info.current_h / 2) + (frame_height / 2)))]

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen.fill((255, 255, 255))

    # Event handler, if esc is pressed the program exits, if space bar is pressed the program pauses/resumes
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                testStopped = True
            if event.key == pygame.K_SPACE:
                testPaused = not testPaused

    # If all the points have been shown pause the program and refresh points
    if not points_left and not drawPoint:
        testPaused = True
        points_left = potential_points
        test_stage += 1

    if testPaused:
        # When the test is paused make the screen grey and calculate timer offset
        screen.fill((100, 100, 100))
        paused_time = pygame.time.get_ticks()
    else:
        elapsed_time = (pygame.time.get_ticks() - paused_time - start_time) / 1000
        # Every 5 seconds
        if round(elapsed_time) % 5 != 0 or elapsed_time < 1:
            # If no point is being displayed pick a new one from the available points
            if not drawPoint:
                drawPoint = True
                test_point = random.choice(points_left)
                points_left.remove(test_point)
                start_error_timer = pygame.time.get_ticks()
        else:
            drawPoint = False
        pygame.draw.circle(screen, (255, 0, 0), test_point, 15)
        pygame.draw.circle(screen, (0, 0, 0), test_point, 5)
    pygame.display.flip()

    return test_point


def image_processing(gray_eye):
    """Performs image processing operations needed in order to extract the pupil. It takes as input a cut-out of an
    eye in gray-scale (gray_eye) and returns a thresholded image."""

    blurred = cv2.GaussianBlur(gray_eye, (3, 3), 0)
    show("Blurred", blurred)
    stretched = cv2.equalizeHist(blurred)  # Contrast stretching with histogram equalization
    show("Stretched", stretched)
    _, threshold_eye = cv2.threshold(stretched, 60, 255, cv2.THRESH_BINARY_INV)
    show("Thresholded", threshold_eye)
    # If the eye is too small, perform Close operation with a smaller kernel
    if gray_eye.shape[0] * gray_eye.shape[1] > 90:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(threshold_eye, cv2.MORPH_CLOSE, kernel)
    show("Closed", closed)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    show("Opened", opened)

    return opened


def get_eye_angle(eye_start_point, eye_end_point):
    """Calculates an eyes' gaze direction. It takes as input the first (eye_start_point) and last (eye_end_point) index
    of the dlib shape predictor referring to the an eye and returns the angle of gaze for that eye. The angle is in
    respect to the camera, hence, looking straight returns an angle of 0, looking left returns a negative value and
    looking right a positive. In the case of an error or of no answer, the return value is False."""

    # Get the region of interest
    temp_coords = np.zeros((landmarks.num_parts, 2), dtype="int")
    for eye_point in range(eye_start_point, eye_end_point + 1):
        temp_coords[eye_point] = (landmarks.part(eye_point).x, landmarks.part(eye_point).y)
    eye_region = temp_coords[eye_start_point: eye_end_point + 1]

    # Find region of interest border
    eye_min_x = np.min(eye_region[:, 0])
    eye_max_x = np.max(eye_region[:, 0])
    eye_min_y = np.min(eye_region[:, 1])
    eye_max_y = np.max(eye_region[:, 1])

    # Crop the region of interest from the frame
    normal_eye = frame[eye_min_y: eye_max_y, eye_min_x: eye_max_x]
    eye_height, eye_width, _ = normal_eye.shape
    show("Normal Eye", normal_eye)

    # Create a mask for only the area in between the eye landmarks
    mask = np.zeros((frame_height, frame_width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 0)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    # Invert the color of the background from black to white
    eye[np.where(mask == [0])] = [255]
    gray_eye = eye[eye_min_y: eye_max_y, eye_min_x: eye_max_x]

    try:
        contour_eye = np.copy(gray_eye)
        contour_eye = cv2.cvtColor(contour_eye, cv2.COLOR_GRAY2BGR)

        # Process image
        gray_eye = image_processing(gray_eye)

        # Find contours in gray-scaled eye image and sort them based on size
        contours, _ = cv2.findContours(gray_eye, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda _: cv2.contourArea(_), reverse=True)
        cv2.drawContours(contour_eye, contours, -1, (0, 0, 255), 1)

        # Choose the largest contour and find the smallest convex set that contains it
        contour = cv2.convexHull(contours[0])
        area = cv2.contourArea(contour)

        # Test contours circularity, if contour is not circular enough ignore it and go to next frame
        circularity = (4 * math.pi * area) / cv2.arcLength(contour, True) ** 2
        if circularity < 0.6 or area < 0.05 * eye_width * eye_height or area > 0.45 * eye_width * eye_height:
            return False

        cv2.ellipse(contour_eye, cv2.fitEllipse(contour), (0, 255, 0), 1)

        # Find the center of the contour (pupil)
        m = cv2.moments(contour)
        if not m['m00'] == 0:
            eye_center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
            if debug_mode != "none":
                cv2.circle(frame, (int(eye_min_x + eye_center[0]), int(eye_min_y + eye_center[1])), 3, (255, 0, 0), -1)

            # Calculate current pupilary displacement in respect to eye center and the corresponding angles
            cornea_displacement_x = (eye_width / 2) - eye_center[0]
            cornea_displacement_y = (eye_height / 2) - eye_center[1]
            angle_x = math.atan(abs(cornea_displacement_x) / to_pixel(10.94, 'mm'))
            angle_y = math.atan(abs(cornea_displacement_y) / to_pixel(10.94, 'mm'))
            if cornea_displacement_x < 0:
                angle_x = -angle_x
            if cornea_displacement_y < 0:
                angle_y = -angle_y
        show("Contours", contour_eye)
    except:
        return False

    # Stabilize angle calculation
    angle_x = stabilize_angle(angle_x, store_eye_angle_x, 0.2, 0.4)
    angle_y = stabilize_angle(angle_y, store_eye_angle_y, 0.1, 0.4)

    return angle_x, angle_y


def get_head_angle():
    """Calculates the head pose. It does not require any input parameters and returns the yaw and pitch of the head in
    respect to the camera. Hence, yaw value is positive when head is turned right and negative when turned left.
    Similarly, pitch value is positive when head is looking upwards and negative when looking downwards"""

    focal_length = frame_width
    camera_matrix = np.array([[focal_length, 0, frame_width / 2], [0, focal_length, frame_height / 2], [0, 0, 1]],
                             dtype="double")

    # Create a list for each landmark coordinate in 2d space and a list for each landmarks corresponding feature
    # coordinate in a 3d model (inspired by: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
    image_points = np.array([(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x, landmarks.part(8).y),
                             (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y),
                             (landmarks.part(48).x, landmarks.part(48).y),
                             (landmarks.part(54).x, landmarks.part(54).y)],
                            dtype="double")
    model_points = np.array([(0, 0, 0), (0, -330, -65), (-225, 170, -135), (225, 170, -135), (-150, -150, -125),
                             (150, -150, -125)], dtype="double")

    # Use the coordinate sets to solve the perspective-n-point problem and acquire head rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                          np.zeros((4, 1), dtype="double"),
                                                          flags=cv2.SOLVEPNP_ITERATIVE)

    if debug_mode != "none":
        projected_point, _ = cv2.projectPoints(np.array([(0, 0, 500)], dtype="double"), rotation_vector,
                                               translation_vector, camera_matrix, np.zeros((4, 1), dtype="double"))
        cv2.line(frame, (int(image_points[0][0]), int(image_points[0][1])),
                 (int(projected_point[0][0][0]), int(projected_point[0][0][1])), (255, 0, 0), 3)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate head pitch and yaw from rotation
    # (inspired by https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    pitch = -math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    if pitch < 0:
        pitch = pitch + math.pi
    else:
        pitch = pitch - math.pi
    yaw = -math.atan2(rotation_matrix[2][0], math.sqrt(rotation_matrix[0][0] ** 2 + rotation_matrix[1][0] ** 2))

    # Stabilize angle calculations
    yaw = stabilize_angle(yaw, store_yaw, 0.1, 0.6)
    pitch = stabilize_angle(pitch, store_pitch, 0.05, 0.4)

    return yaw, pitch


def get_gaze_coords():
    """Determines the coordinates of where a person is looking by combining the head pose and gaze direction. It does
    not require any input parameters and returns the coordinates at which the person is looking as well as the X and Y
    angles of the eyes and head respectively."""

    global previous_x_result, previous_x_error, previous_y_result, previous_y_error
    left_eye_edges = 36, 41
    right_eye_edges = 42, 47
    middle_face_point = 33
    # Get origin from where calculation are going to be performed
    base_x = landmarks.part(middle_face_point).x - (frame_width / 2)
    base_y = landmarks.part(27).y - (frame_height / 2)

    # Get eye angles for both eyes
    left_eye_angle = get_eye_angle(left_eye_edges[0], left_eye_edges[1])
    right_eye_angle = get_eye_angle(right_eye_edges[0], right_eye_edges[1])

    # Get one angle to represent both eyes. If we successfully detected both eyes then an average will be used. In the
    # case that the two angles detected are significantly different then we will give more power to the smaller
    # calculation
    if left_eye_angle and right_eye_angle:
        if abs(right_eye_angle[0] - left_eye_angle[0]) > 0.05:
            if abs(right_eye_angle[0]) < abs(left_eye_angle[0]):
                if abs(right_eye_angle[0]) < math.pi / 18:
                    eye_angle_x = right_eye_angle[0]
                else:
                    eye_angle_x = (right_eye_angle[0] + 2 * left_eye_angle[0]) / 3
            else:
                if abs(right_eye_angle[0]) < math.pi / 18:
                    eye_angle_x = left_eye_angle[0]
                else:
                    eye_angle_x = (2 * right_eye_angle[0] + left_eye_angle[0]) / 3
        else:
            eye_angle_x = (right_eye_angle[0] + left_eye_angle[0]) / 2
        eye_angle_x -= 0.06
        eye_angle_y = (right_eye_angle[1] + left_eye_angle[1]) / 2
    elif left_eye_angle:
        eye_angle_x = left_eye_angle[0]
        eye_angle_y = left_eye_angle[1]
    elif right_eye_angle:
        eye_angle_x = right_eye_angle[0]
        eye_angle_y = right_eye_angle[1]
    else:
        eye_angle_x = False
        eye_angle_y = False
    eye_angle_y = 1.5 * (eye_angle_y - 0.1)

    # Get head angle
    head_angle_x, head_angle_y = get_head_angle()

    if debug_mode != "none":
        cv2.line(frame, (int((frame_width / 2) + (frame_width / 4)), 0), (int((frame_width / 2) + (frame_width / 4)),
                                                                          frame_height), (0, 0, 255), 2)
        cv2.line(frame, (int((frame_width / 2) - (frame_width / 4)), 0), (int((frame_width / 2) - (frame_width / 4)),
                                                                          frame_height), (0, 0, 255), 2)
        cv2.line(frame, (int(frame_width / 2), 0), (int(frame_width / 2), frame_height), (0, 0, 255), 3)
        cv2.line(frame, (0, int(frame_height / 2)), (frame_width, int(frame_height / 2)), (0, 0, 255), 3)

    # Calculate the coordinates of where the head is facing
    x_result = base_x + (distance * -math.degrees(math.tan(math.radians(head_angle_x))))
    y_result = base_y + (distance * -math.degrees(math.tan(math.radians(head_angle_y))))

    if debug_mode == "full" or debug_mode == "angles":
        cv2.line(frame, (int(x_result + (frame_width / 2)), 0), (int(x_result + (frame_width / 2)), frame_height),
                 (0, 255, 0), 3)
        cv2.line(frame, (0, int(y_result + (frame_height / 2))), (frame_width, int(y_result + (frame_height / 2))),
                 (0, 255, 0), 3)

    # If an eye angle has been detected use it to enhance the coordinate calculation
    if eye_angle_x and eye_angle_y:

        global previous_eye_x, previous_eye_x_error, previous_eye_y, previous_eye_y_error

        # Use Kalman filtering to reduce noise in eye angles
        previous_eye_x, previous_eye_x_error = kalman(abs(eye_angle_x), previous_eye_x, previous_eye_x_error, 0.1)
        if eye_angle_x < 0:
            eye_angle_x = -previous_eye_x
        else:
            eye_angle_x = previous_eye_x
        previous_eye_y, previous_eye_y_error = kalman(abs(eye_angle_y), previous_eye_y, previous_eye_y_error, 0.05)
        if eye_angle_y < 0:
            eye_angle_y = -previous_eye_y
        else:
            eye_angle_y = previous_eye_y

        # Calculate the coordinates where the eyes are looking
        eye_disp_x = base_x + distance * -math.degrees(math.tan(math.radians(eye_angle_x)))
        eye_disp_y = base_y + (distance * -math.degrees(math.tan(math.radians(eye_angle_y))))

        if debug_mode == "full" or debug_mode == "angles":
            cv2.line(frame, (int(eye_disp_x + (frame_width / 2)), 0),
                     (int(eye_disp_x + (frame_width / 2)), frame_height), (255, 255, 0), 3)
            cv2.line(frame, (0, int(eye_disp_y + (frame_height / 2))),
                     (frame_width, int(eye_disp_y + (frame_height / 2))), (255, 255, 0), 3)

        # Calculate the X coordinate when combining both the head pose and the eye angle
        kx_head_origin = (abs(x_result - base_x) - (2 * base_x * x_result / frame_width)) / frame_width
        kx_eye_head = (abs(eye_disp_x - x_result) - (2 * x_result * eye_disp_x / frame_width)) / frame_width
        x_result = (1 + kx_head_origin + kx_eye_head) * eye_disp_x + (kx_head_origin - kx_eye_head) * x_result
        previous_x_result, previous_x_error = kalman(x_result, previous_x_result, previous_x_error, 10)
        x_result = previous_x_result

        # Calculate the Y coordinate when combining both the head pose and the eye angle
        ky_head_origin = (abs(y_result - base_y) - (2 * base_y * y_result / frame_height)) / frame_height
        ky_eye_head = (abs(eye_disp_y - y_result) - (2 * y_result * eye_disp_y / frame_height)) / frame_height
        y_result = (1 + ky_head_origin + ky_eye_head) * eye_disp_y + (ky_head_origin - ky_eye_head) * y_result
        previous_y_result, previous_y_error = kalman(y_result, previous_y_result, previous_y_error, 5)
        y_result = previous_y_result

        if debug_mode != "none":
            cv2.line(frame, (int(x_result + (frame_width / 2)), 0), (int(x_result + (frame_width / 2)), frame_height),
                     (255, 0, 0), 3)
            cv2.line(frame, (0, int(y_result + (frame_height / 2))), (frame_width, int(y_result + (frame_height / 2))),
                     (255, 0, 0), 3)

    # Restore the value offset that happened because of shifting the origin before returning
    return (-x_result + (frame_width / 2), -y_result + (frame_height / 2)), (eye_angle_x, eye_angle_y), \
           (head_angle_x, head_angle_y)


while True:
    if isPepper:
        frame = connect.get_img()
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
    else:
        success, frame = connect.read()
        if not success:
            break

    frame_height, frame_width, _ = frame.shape
    start = time.perf_counter()
    frame_num += 1
    distance = to_pixel(real_distance)

    # Convert frame to grayscale and attempt to detect a face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame, 0)

    for face in faces:
        # For every face detected, calculate the landmarks
        landmarks = shape_predictor(gray_frame, face)

        if debug_mode != "none":
            landmark_coords = np.zeros((landmarks.num_parts, 2), dtype="int")
            for i in range(0, landmarks.num_parts):
                landmark_coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
            for landmark_coord in landmark_coords:
                cv2.circle(frame, landmark_coord, 1, (0, 0, 255), -1)

        # Get gaze direction
        coord_estimate, eye_angle, head_angle = get_gaze_coords()

        if testMode and not testStopped:
            # Run tests
            point_tested = run_test()

            # Adjust points so they give the same coordinates on different sized screens
            if point_tested:
                if point_tested[0] == ((screen_info.current_w / 2) - (frame_width / 2)):
                    point_tested = (int((frame_width / 2) - (frame_width / 4)), frame_height / 2)
                elif point_tested[0] == ((screen_info.current_w / 2) + (frame_width / 2)):
                    point_tested = (int((frame_width / 2) + (frame_width / 4)), frame_height / 2)
                elif point_tested[1] == ((screen_info.current_h / 2) - (frame_height / 2)):
                    point_tested = (frame_width / 2, int((frame_height / 2) - (frame_height / 4)))
                elif point_tested[1] == ((screen_info.current_h / 2) + (frame_height / 2)):
                    point_tested = (frame_width / 2, int((frame_height / 2) + (frame_height / 4)))
                else:
                    point_tested = (frame_width / 2, frame_height / 2)
            if not testPaused and (pygame.time.get_ticks() - start_error_timer) / 1000 > 2.5:
                # Store test results
                test_results.append([test_stage, int(point_tested[0]), int(point_tested[1]),
                                     coord_estimate[0], coord_estimate[1], round(eye_angle[0], 3),
                                     round(eye_angle[1], 3), round(head_angle[0], 3), round(head_angle[1], 3)])

    print("Time taken:", round(time.perf_counter() - start, 5), "Effective fps:",
          round(1 / (time.perf_counter() - start)), "Frame:", frame_num)
    show('frame', frame, 1)

    key = cv2.waitKey(30)
    if key == 27 or testStopped:
        break

if testMode:
    pygame.display.quit()
    pygame.quit()
if isPepper:
    connect.close_connection()
if test_results:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = None
    df = pd.DataFrame(test_results)
    df.columns = ['S.', 'ExpX', 'ExpY', 'EstimatedX', 'EstimatedY', 'EyeX', 'EyeY', 'HeadX', 'HeadY']
    df.to_pickle("test_" + str(test_number) + "_" + test_name + "_" + str(real_distance))
    print(df)
cv2.destroyAllWindows()
