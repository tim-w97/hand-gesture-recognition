from hand_recognizer import recognize_all
import cv2


def recognize_capture(numpy_image):
    json = recognize_all(numpy_image)

    print('+++ JSON Result +++')
    print(json)


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    flipped_image = cv2.flip(frame, 1)

    # Display instructions on how to capture an image.
    cv2.putText(
        flipped_image,
        "Press 'c' to capture an image",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow('Webcam', flipped_image)

    key = cv2.waitKey(1) & 0xFF

    # Check for 'c' key press to capture an image
    if key == ord('c'):
        recognize_capture(flipped_image)
        break

    # end loop when user presses ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
