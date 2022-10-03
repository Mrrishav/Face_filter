import dlib
import cv2
import numpy as np
from math import hypot
from scipy.spatial import distance

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("pig_nose.png")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        landmarks = predictor(gray_frame,face)
        top_nose = (landmarks.part(29).x,landmarks.part(29).y)
        center_nose = (landmarks.part(30).x,landmarks.part(30).y)
        left_nose = (landmarks.part(31).x,landmarks.part(31).y)
        right_nose = (landmarks.part(35).x,landmarks.part(35).y)

        # origin = (landmarks.part(0).x, landmarks.part(0).y)
        # # print("origin",origin)
        # end = (landmarks.part(16).x, landmarks.part(16).y)
        # # print("end",end)
        # mid = (landmarks.part(27).x, landmarks.part(27).y)
        # eye = (landmarks.part(19).x, landmarks.part(19).y)
        # ms = (landmarks.part(29).x, landmarks.part(29).y)
        #
        # dist_hor = distance.euclidean(origin,end)
        # dist_ver = distance.euclidean([ms[0],eye[1]],ms)
        # print("Horizontal distance",dist_ver)

        # start_point = (241, 239)
        # end_point = (437, 238)
        # color = (255,0,0)
        # cv2.line(frame, start_point, end_point, color, 1)
        nose_width = int(hypot(left_nose[0]-right_nose[0],left_nose[1]-right_nose[1])*1.7)
        # print(nose_width)
        nose_height = int(nose_width * 0.77)
        # cv2.rectangle(frame,(int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2)),(int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2)),(255,0,0),2)
        top_left = (int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2))
        bottom_right = (int(center_nose[1]-nose_width/2),int(center_nose[1]-nose_height/2))
        nose_png = cv2.resize(nose_image, (nose_width, nose_height))
        nose_area = frame[top_left[1]: top_left[1] + nose_height, top_left[0]:top_left[0]+nose_width]

        nose_pig_gray = cv2.cvtColor(nose_png, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray,35,255,cv2.THRESH_BINARY_INV)
        # _, show = cv2.threshold(spec_dim_gray, 36, 255, cv2.THRESH_BINARY_INV)
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)

        final_nose = cv2.add(nose_area_no_nose,nose_png)
        frame[top_left[1]: top_left[1] + nose_height, top_left[0]:top_left[0] + nose_width] = final_nose
    cv2.imshow("Frame", frame)
    # cv2.imshow("Pig nose",nose_image)
    key = cv2.waitKey(1)
    if key == 27:
        break