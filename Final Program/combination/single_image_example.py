from hand_recognizer import recognize_all
from visualization import add_visualization
import cv2
import pprint
import json


def recognize_capture(numpy_image):
    json_str = recognize_all(numpy_image)

    print()

    print('# # # # # # # #')
    print('# JSON Result #')
    print('# # # # # # # #')

    print()

    pprint.pprint(
        json.loads(json_str),
        compact=True
    )

    image_with_visualization = add_visualization(
        numpy_image,
        json_str
    )

    cv2.imshow('Result', image_with_visualization)
    cv2.waitKey(0)


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    flipped_image = cv2.flip(frame, 1)

    annotated_image = flipped_image.copy()

    # Display instructions on how to capture an image.
    cv2.putText(
        annotated_image,
        "Press 'c' to capture an image",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow('Webcam', annotated_image)

    key = cv2.waitKey(1) & 0xFF

    # Check for 'c' key press to capture an image
    if key == ord('c'):
        cap.release()
        cv2.destroyAllWindows()

        recognize_capture(flipped_image)
        break

    # end loop when user presses ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
