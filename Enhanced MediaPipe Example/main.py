import mediapipe as mp
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

showed_text = ''
cap = None

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global showed_text
    showed_text = ''

    for gesture in result.gestures:
        showed_text += str(gesture[0].category_name)
        showed_text += ' + '

    showed_text = showed_text[:len(showed_text) - 3]

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=3
)

def start_recognition():
    global cap
    cap = cv2.VideoCapture(1)
    with GestureRecognizer.create_from_options(options) as recognizer:
        process_webcam_video(cap, recognizer)

def stop_recognition():
    global cap
    if cap is not None:
        cap.release()
        root.destroy()

def process_webcam_video(cap, recognizer):
    timestamp = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Got no video stream!")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_image, timestamp)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, showed_text, (10, 100), font, 4, (255, 255, 255), 5, cv2.LINE_AA)

        cv2.imshow('image', frame)

        timestamp += 1

        if cv2.waitKey(1) == ord('q'):
            break

def resize_background(event):
    global background_image, background_label
    # Get the window size
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Resize the background image to match the window size
    resized_background = background_image.copy()
    resized_background.thumbnail((window_width, window_height))
    background_label.config(image=resized_background)
    background_label.photo = resized_background  # Keep a reference to prevent image from being garbage collected

root = tk.Tk()
root.title("Gesture Recognition")

# Load and set a custom background image with a specified size
background_image = Image.open("3.jpeg")
background_image = ImageTk.PhotoImage(background_image)

background_width = 800  # Set the width for the background image
background_height = 600  # Set the height for the background image

frame = ttk.Frame(root)
frame.pack(fill='both', expand=True)

root.geometry(f"{background_width}x{background_height}")  # Set the window size

background_label = tk.Label(root, image=background_image, width=background_width, height=background_height)
background_label.place(relx=0.5, rely=0.5, anchor="center")
background_label.bind('<Configure>', resize_background)

# Create more beautiful buttons with specified dimensions
start_img = Image.open("start.webp")
start_img = start_img.resize((200, 250))  # Set the dimensions for the start button image
start_img = ImageTk.PhotoImage(start_img)

stop_img = Image.open("stop.png")
stop_img = stop_img.resize((160, 60))  # Set the dimensions for the stop button image
stop_img = ImageTk.PhotoImage(stop_img)

def start_recognition_click():
    start_recognition()  # Call the start_recognition function

def stop_recognition_click():
    stop_recognition()  # Call the stop_recognition function

start_button = tk.Button(root, image=start_img, command=start_recognition_click, borderwidth=0, width=150, height=50)
stop_button = tk.Button(root, image=stop_img, command=stop_recognition_click, borderwidth=0, width=150, height=50)

start_button.place(relx=0.2, rely=0.8, anchor="center")
stop_button.place(relx=0.8, rely=0.8, anchor="center")

# Prevent the buttons from being garbage collected
start_button.image = start_img
stop_button.image = stop_img

root.mainloop()
