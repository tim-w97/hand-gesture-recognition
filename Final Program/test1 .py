import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import json

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands):
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Check if landmarks are found and draw them.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255),
                                                                                      thickness=2, circle_radius=2))
    return output_image, results

def countFingers(image, results):
    height, width, _ = image.shape
    output_image = image.copy()
    count = {'RIGHT': 0, 'LEFT': 0, }
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            count[hand_label.upper()] += 1

    return output_image, fingers_statuses, count

# Read the input image.
image = cv2.imread('media/sample1.jpg')

# Perform Hands landmarks detection on the image.
image_with_landmarks, hands_results = detectHandsLandmarks(image, hands)

# Count the number of fingers in the image.
image_with_finger_count, fingers_statuses, count = countFingers(image_with_landmarks, hands_results)

# Display the image with finger count.
plt.imshow(image_with_finger_count[:, :, ::-1])
plt.title("Image with Finger Count")
plt.axis('off')
plt.show()

# Output the finger count as JSON.
output_json = {'count': count}
with open('finger_count.json', 'w') as json_file:
    json.dump(output_json, json_file)
