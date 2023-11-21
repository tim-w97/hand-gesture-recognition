from hand_recognizer import recognize_all
from visualization import add_visualization
import cv2


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    flipped_image = cv2.flip(frame, 1)

    json_result = recognize_all(flipped_image)

    image_with_visualization = add_visualization(flipped_image, json_result)

    cv2.imshow('Webcam', image_with_visualization)

    key = cv2.waitKey(1) & 0xFF

    # end loop when user presses ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
