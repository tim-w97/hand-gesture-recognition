import hand_recognizer
import cv2


image_path = 'test_images/test_1.jpg'  # Replace with the path to your image file
image_np = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

json = hand_recognizer.recognize_all(
  numpy_image=image_rgb
)

print(json)
