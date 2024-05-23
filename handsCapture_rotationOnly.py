import time
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
handLmStyle = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=10)

# Parameters for finger 0
prev_x_position1 = 0
cur_x_position1 = 0
prev_y_position1 = 0
cur_y_position1 = 0

rotate_positive_timer = 0
rotate_negative_timer = 0

# Parameters for finger 8
x_position_8 = 0
y_position_8 = 0

# Parameters for finger 4
x_position_4 = 0
y_position_4 = 0

# Pinch parameter
pinch_timer = 0

def capture_hands(hand):
    return hands.process(hand)

def draw(img, handLms):
    mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS, handLmStyle,
                                              handConStyle)

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

while True:
    startTime = time.time()
    success, img_0 = cap.read()
    if not success:
        continue

    img = cv2.flip(img_0, 1)

    # Obtain image dimensions
    imgHeight, imgWidth, _ = img.shape

    # Convert the image to RGB format
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = capture_hands(img_RGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            draw(img, handLms)

        for index, lm in enumerate(handLms.landmark):
            # Get coordinates and append to the list
            x = int(imgWidth * lm.x)
            y = int(imgHeight * lm.y)
            lm_list.append([x, y])
            # Add index labels
            cv2.putText(img, str(index), (x - 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # Construct convex hull points
        lm_array = np.array(lm_list, dtype=np.int32)
        hull_index = [0, 1, 2, 3, 6, 10, 14, 17, 18]
        hull = cv2.convexHull(lm_array[hull_index, :])
        cv2.polylines(img, [hull], True, (0, 255, 0), 2)

        # Find the outer points
        ll = [4, 8, 12, 16, 20]
        outer_finger_index = [i for i in ll if
                              cv2.pointPolygonTest(hull, (lm_list[i][0], lm_list[i][1]), True) < -10]

        # Get gesture
        gesture = get_gesture(outer_finger_index)

        # Determine the relationship between finger 4 and finger 8 (pinch detection)
        x_position_8 = handLms.landmark[8].x
        x_position_4 =  handLms.landmark[4].x
        y_position_8 = handLms.landmark[8].y
        y_position_4 = handLms.landmark[4].y
        distance = ((x_position_8 - x_position_4) ** 2 + (y_position_8 - y_position_4) ** 2) ** 0.5
        if distance < 0.08:
            pinch_timer = 10
        else:
            pinch_timer -= 1

        if pinch_timer > 0:
            cv2.putText(img, str("pinch"), (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        # Determine if finger 8 moves left, right, up or down
        if prev_x_position1 == 0 or prev_y_position1 == 0:
            prev_x_position1 = handLms.landmark[8].x
            prev_y_position1 = handLms.landmark[8].y
        else:
            cur_x_position1 = handLms.landmark[8].x
            cur_y_position1 = handLms.landmark[8].y
            # Calculate displacement on x-axis
            if (cur_x_position1 - prev_x_position1) > 0.07 and (cur_y_position1 - prev_y_position1) > 0.07:
                rotate_positive_timer = 10
                rotate_negative_timer = 0
            elif (cur_x_position1 - prev_x_position1) < -0.07 and (cur_y_position1 - prev_y_position1) > 0.07:
                rotate_positive_timer = 0
                rotate_negative_timer = 10
            prev_x_position1 = cur_x_position1
            prev_y_position1 = cur_y_position1

    # Determine rotation direction
    if rotate_positive_timer > 0 and rotate_negative_timer == 0 :
        cv2.putText(img, str("Rotate positive"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        rotate_positive_timer -= 1
    elif rotate_positive_timer == 0 and rotate_negative_timer > 0:
        cv2.putText(img, str("Rotate negative"), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        rotate_negative_timer -= 1

    # Add FPS
    endTime = time.time()
    fps = 1 / (endTime - startTime)
    cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
