import mediapipe as mp
from mediapipe.tasks import python
import cv2
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

showed_text = ''


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global showed_text

    for gesture in result.gestures:
        showed_text = str(gesture[0].category_name)


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=3
)

cap = cv2.VideoCapture(0)


def process_webcam_video():
    timestamp = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Got no video stream!")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_image, timestamp)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, showed_text, (10, 100), font, 5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('image', frame)

        timestamp += 1

        if cv2.waitKey(1) == ord('q'):
            break


with GestureRecognizer.create_from_options(options) as recognizer:
    process_webcam_video()

cap.release()
