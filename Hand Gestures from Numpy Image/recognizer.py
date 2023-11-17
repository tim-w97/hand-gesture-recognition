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


# recognizes gestures from an image (number array) and returns them as a json
def recognize_gestures(numpy_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    recognition_result = recognizer.recognize(mp_image)

    gestures = []

    for gesture_list in recognition_result.gestures:
        if len(gesture_list) == 0:
            continue

        gestures.append(gesture_list[0].category_name)

    result = {
      "gestures": gestures
    }

    return json.dumps(result)
