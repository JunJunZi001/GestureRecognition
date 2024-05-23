import time
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
handLmStyle = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=10)

# parameters for finger0
prev_x_position1 = 0
cur_x_position1 = 0
prev_y_position1 = 0
cur_y_position1 = 0

# parameters for finger12
prev_y_position2 = 0
cur_y_position2 = 0
waving_hand = False
waving_hand_timer = 0

moving_right_timer = 0
moving_left_timer = 0

moving_up_timer = 0
moving_down_timer = 0


def get_gesture(outer_finger_index):
    if len(outer_finger_index) == 0:
        return "0"
    elif len(outer_finger_index) == 1 and outer_finger_index[0] == 8:
        return "1"
    elif len(outer_finger_index) == 2 and (outer_finger_index[0] == 8 and outer_finger_index[1] == 12):
        return "2"
    elif len(outer_finger_index) == 3 and (outer_finger_index[0] == 8 and outer_finger_index[1] == 12
                                           and outer_finger_index[2] == 16):
        return "3"
    elif len(outer_finger_index) == 4 and (outer_finger_index[0] == 8 and outer_finger_index[1] == 12
                                           and outer_finger_index[2] == 16 and outer_finger_index[3] == 20):
        return "4"
    elif len(outer_finger_index) == 5 and (outer_finger_index[0] == 4 and outer_finger_index[1] == 8
                                           and outer_finger_index[2] == 12 and outer_finger_index[3] == 16 and
                                           outer_finger_index[4] == 20):
        return "5"
    else:
        return None

# Use loop to make the video a series of pictures it takes
while True:
    startTime = time.time()
    success, img_0 = cap.read()
    if not success:
        continue

    # flip the image to make it easier to adjustment
    img = cv2.flip(img_0, 1)

    # introduce the length, width and height of the picture
    imgHeight, imgWidth, _ = img.shape

    # turn it to RGB forms
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read and process the image
    result = hands.process(img_RGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS,
                                                      handLmStyle, handConStyle)

            for index, lm in enumerate(handLms.landmark):
                # get coordinates and append to the coordinate point sequence
                x = int(imgWidth * lm.x)
                y = int(imgHeight * lm.y)
                lm_list.append([x, y])
                # add numbering
                cv2.putText(img, str(index), (x - 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # construct convex hull points
            lm_array = np.array(lm_list, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 17, 18]
            hull = cv2.convexHull(lm_array[hull_index, :])
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            # find outer points
            ll = [4, 8, 12, 16, 20]
            outer_finger_index = [i for i in ll if
                                  cv2.pointPolygonTest(hull, (lm_list[i][0], lm_list[i][1]), True) < -10]

            # get gesture
            gesture = get_gesture(outer_finger_index)

            # annotate recognized gestures
            if gesture is not None:
                cv2.putText(img, gesture, (90, 90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 255, 0), 4)
                if gesture == "2":
                    cv2.putText(img, "voice assistant", (90, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                elif gesture == "1":
                    cv2.putText(img, "select", (90, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                elif gesture == "0":
                    cv2.putText(img, "back", (90, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

            # determine left and right movement of finger 0
            if prev_x_position1 == 0:
                prev_x_position1 = handLms.landmark[0].x
            else:
                cur_x_position1 = handLms.landmark[0].x
                # calculate displacement on the x-axis
                if (cur_x_position1 - prev_x_position1) > 0.07:
                    moving_right_timer = 10
                    moving_left_timer = 0
                elif moving_right_timer != 10 and (cur_x_position1 - prev_x_position1) < -0.07:
                    moving_left_timer = 10
                    moving_right_timer = 0
                else:
                    root_x_Move = False;
                prev_x_position1 = cur_x_position1

            # determine up and down movement of finger 0
            if prev_y_position1 == 0:
                prev_y_position1 = handLms.landmark[0].y
            else:
                cur_y_position1 = handLms.landmark[0].y
                # calculate displacement on the y-axis
                if (cur_y_position1 - prev_y_position1) > 0.07:
                    moving_down_timer = 10
                    moving_up_timer = 0
                elif moving_down_timer != 10 and (cur_y_position1 - prev_y_position1) < -0.07:
                    moving_up_timer = 10
                    moving_down_timer = 0
                else:
                    root_y_Move = False;
                prev_y_position1 = cur_y_position1

            # determine up and down movement of finger 12
            if prev_y_position2 == 0:
                prev_y_position2 = handLms.landmark[12].y
            else:
                cur_y_position2 = handLms.landmark[12].y
                # calculate displacement on the y-axis
                if (cur_y_position2 - prev_y_position2) > 0.07 and moving_down_timer == 0 and moving_up_timer == 0:
                    waving_hand_timer = 10
                elif (cur_y_position2 - prev_y_position2) < -0.07 and moving_down_timer == 0 and moving_up_timer == 0:
                    waving_hand_timer = 10
                prev_y_position2 = cur_y_position2

    # determine left and right movement
    if moving_right_timer > 0 and moving_left_timer == 0:
        cv2.putText(img, str("Moving right"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        if gesture == "5":
         cv2.putText(img, str("ON"), (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        moving_right_timer -= 1
    elif moving_left_timer > 0 and moving_right_timer == 0:
        cv2.putText(img, str("Moving left"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        if gesture == "5":
         cv2.putText(img, str("OFF"), (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        moving_left_timer -= 1
    # determine up and down movement
    elif moving_down_timer > 0 and moving_up_timer == 0:
        cv2.putText(img, str("Moving down"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        moving_down_timer -= 1
    elif moving_up_timer > 0 and moving_down_timer == 0:
        cv2.putText(img, str("Moving up"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        moving_up_timer -= 1
    # determine waving hand gesture
    elif waving_hand_timer > 0:
        cv2.putText(img, str("Waving hand"), (90, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        waving_hand_timer -= 1

    # add FPS
    # startTime has been defined above.
    endTime = time.time()
    fps = 1 / (endTime - startTime)
    cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # display the image
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
