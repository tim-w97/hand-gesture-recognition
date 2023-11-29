import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import json
import os

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=6, min_detection_confidence=0.5)

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
    count = {'RIGHT': 0, 'LEFT': 0}
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

# Create a directory to save images and JSON files.
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Variable to indicate if an image is captured.
image_captured = False

while camera_video.isOpened():
    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Display instructions on how to capture an image.
    cv2.putText(frame, "Press 'c' to capture an image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame.
    cv2.imshow('Fingers Counter', frame)

    # Wait for key events.
    key = cv2.waitKey(1) & 0xFF

    # Check for 'c' key press to capture an image.
    if key == ord('c'):
        # Set the flag to indicate that an image is captured.
        image_captured = True

    # Check if an image is captured.
    if image_captured:
        # Perform Hands landmarks detection on the captured image.
        image_with_landmarks, hands_results = detectHandsLandmarks(frame, hands)

        # Count the number of fingers in the image.
        image_with_finger_count, fingers_statuses, count = countFingers(image_with_landmarks, hands_results)

        # Display the image with finger count.
        plt.imshow(image_with_finger_count[:, :, ::-1])
        plt.title("Image with Finger Count")
        plt.axis('off')
        plt.show()

        # Save the image with finger count.
        output_image_path = os.path.join(output_dir, 'captured_image_with_finger_count.jpg')
        cv2.imwrite(output_image_path, image_with_finger_count)

        # Output the finger count as JSON.
        output_json_path = os.path.join(output_dir, 'finger_count.json')
        output_json = {'fingers_statuses': fingers_statuses, 'count': count}
        with open(output_json_path, 'w') as json_file:
            json.dump(output_json, json_file)

        print(f"Image with finger count saved at: {output_image_path}")
        print(f"Finger count JSON saved at: {output_json_path}")

        # Reset the flag.
        image_captured = False

    # Check if 'ESC' is pressed and break the loop.
    elif key == 27:
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
