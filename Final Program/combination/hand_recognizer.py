import mediapipe as mp
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path='gesture_recognizer.task'
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=5
)

recognizer = vision.GestureRecognizer.create_from_options(options)


# recognize both gestures and fingers count and return a json
def recognize_all(numpy_image):
    gestures = recognize_gestures(numpy_image)
    fingers_amount = count_fingers(numpy_image)

    result = {
        "gestures": gestures,
        "totalFingersAmount": fingers_amount
    }

    return json.dumps(result)


# recognizes gestures from an image (number array) and returns them as a json
def recognize_gestures(numpy_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    recognition_result = recognizer.recognize(mp_image)

    gestures = []

    for idx, gesture_list in enumerate(recognition_result.gestures):
        if len(gesture_list) == 0:
            continue

        gesture_name = gesture_list[0].category_name

        if gesture_name == "None":
            gestures.append("Unknown Gesture")
            continue

        single_hand_landmarks = recognition_result.hand_landmarks[idx]
        middle_finger_mcp = single_hand_landmarks[9]

        gesture_info = {
            "name": gesture_name,
            "pos_x": middle_finger_mcp.x,
            "pos_y": middle_finger_mcp.y,
        }

        gestures.append(gesture_info)

    return gestures


# counts the fingers on an image (number array) and returns the result as a json
def count_fingers(numpy_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    recognition_result = recognizer.recognize(mp_image)

    fingers_amount = 0

    # there could be multiple hands with each their landmarks, so we have to create a loop
    for single_hand_landmarks in recognition_result.hand_landmarks:
        fingers_amount += count_fingers_on_single_hand(single_hand_landmarks)

    return fingers_amount


# counts the fingers of a single hand
def count_fingers_on_single_hand(single_hand_landmarks):
    wrist = single_hand_landmarks[0]
    thumb_cmc = single_hand_landmarks[1]
    thumb_mcp = single_hand_landmarks[2]
    thumb_ip = single_hand_landmarks[3]
    thumb_tip = single_hand_landmarks[4]
    index_finger_mcp = single_hand_landmarks[5]
    index_finger_pip = single_hand_landmarks[6]
    index_finger_dip = single_hand_landmarks[7]
    index_finger_tip = single_hand_landmarks[8]
    middle_finger_mcp = single_hand_landmarks[9]
    middle_finger_pip = single_hand_landmarks[10]
    middle_finger_dip = single_hand_landmarks[11]
    middle_finger_tip = single_hand_landmarks[12]
    ring_finger_mcp = single_hand_landmarks[13]
    ring_finger_pip = single_hand_landmarks[14]
    ring_finger_dip = single_hand_landmarks[15]
    ring_finger_tip = single_hand_landmarks[16]
    pinky_mcp = single_hand_landmarks[17]
    pinky_pip = single_hand_landmarks[18]
    pinky_dip = single_hand_landmarks[19]
    pinky_tip = single_hand_landmarks[20]

    count = 0

    # index finger is up
    if index_finger_tip.y < index_finger_dip.y:
        count += 1

    # middle finger is up
    if middle_finger_tip.y < middle_finger_dip.y:
        count += 1

    # ring finger is up
    if ring_finger_tip.y < ring_finger_dip.y:
        count += 1

    # pinky is up
    if pinky_tip.y < pinky_dip.y:
        count += 1

    # thumb is up (special case)

    # if it's the thumb of the left hand
    if thumb_tip.x < pinky_tip.x and thumb_tip.x < index_finger_mcp.x:
        count += 1

    # if it's the thumb of the right hand
    if thumb_tip.x > pinky_tip.x and thumb_tip.x > index_finger_mcp.x:
        count += 1

    return count
