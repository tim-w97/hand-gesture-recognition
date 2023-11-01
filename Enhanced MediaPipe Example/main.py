import mediapipe as mp
from mediapipe.tasks import python
import cv2
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print(result.gestures)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=3
)

cap = cv2.VideoCapture(0)

frame_timestamp_ms = 0

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Got no video stream!")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        cv2.imshow("frame", frame)

        frame_timestamp_ms += 1

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
