import cv2
import hand_recognizer

cap = cv2.VideoCapture(0)


def draw_text_on_image(image, text):
    cv2.putText(
        img=image,
        text=text,
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,  # Corrected argument name
        color=(0, 255, 0),  # Renamed font_color to color (optional)
        thickness=2
    )

    return image


while True:
    ret, frame = cap.read()

    flipped_image = cv2.flip(frame, 1)

    json = hand_recognizer.recognize_all(numpy_image=flipped_image)

    output_image = draw_text_on_image(flipped_image, json)

    cv2.imshow('Webcam', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
