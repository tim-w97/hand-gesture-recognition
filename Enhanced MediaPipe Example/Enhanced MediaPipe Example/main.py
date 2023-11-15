import mediapipe as mp
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
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

image_mode_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
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


def process_image_file(file_path, recognizer):
    try:
        image = cv2.imread(file_path)
        if image is not None:
            print("Image loaded successfully.")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = recognizer.recognize(mp_image)

            if result.gestures:
                showed_text = ''
                for gesture in result.gestures:
                    showed_text += f"{gesture[0].category_name} "

                print(f"Detected Gestures: {showed_text}")

                cv2.putText(image, showed_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)
            else:
                print("No gestures detected.")

            cv2.imshow('image', image)
            cv2.waitKey(0)
        else:
            print("Error loading image.")
    except Exception as e:
        print(f"Error processing image: {e}")





def open_image():
    file_path = filedialog.askopenfilename(title="Select Image File")
    if file_path:
        print("Selected file:", file_path)
        try:
            with GestureRecognizer.create_from_options(image_mode_options) as recognizer:
                process_image_file(file_path, recognizer)
        except Exception as e:
            print(f"Error processing image: {e}")





def resize_background(event):
    global background_image, background_label
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    resized_background = background_image.copy()
    resized_background.thumbnail((window_width, window_height))
    background_label.config(image=resized_background)
    background_label.photo = resized_background


root = tk.Tk()
root.title("Gesture Recognition")

background_image = Image.open("3.jpeg")
background_image = ImageTk.PhotoImage(background_image)

background_width = 800
background_height = 600

frame = ttk.Frame(root)
frame.pack(fill='both', expand=True)

root.geometry(f"{background_width}x{background_height}")

background_label = tk.Label(root, image=background_image, width=background_width, height=background_height)
background_label.place(relx=0.5, rely=0.5, anchor="center")
background_label.bind('<Configure>', resize_background)

start_img = Image.open("start.webp")
start_img = start_img.resize((200, 250))
start_img = ImageTk.PhotoImage(start_img)

stop_img = Image.open("stop.png")
stop_img = stop_img.resize((160, 60))
stop_img = ImageTk.PhotoImage(stop_img)

open_img = Image.open("open.webp")
open_img = open_img.resize((160, 60))
open_img = ImageTk.PhotoImage(open_img)

options.running_mode = VisionRunningMode.LIVE_STREAM


def start_recognition_click():
    options.running_mode = VisionRunningMode.LIVE_STREAM
    start_recognition()


def stop_recognition_click():
    stop_recognition()


def open_image_click():
    open_image()


start_button = tk.Button(root, image=start_img, command=start_recognition_click, borderwidth=0, width=150, height=50)
stop_button = tk.Button(root, image=stop_img, command=stop_recognition_click, borderwidth=0, width=150, height=50)
open_button = tk.Button(root, image=open_img, command=open_image_click, borderwidth=0, width=150, height=50)

start_button.place(relx=0.2, rely=0.8, anchor="center")
stop_button.place(relx=0.8, rely=0.8, anchor="center")
open_button.place(relx=0.5, rely=0.8, anchor="center")

start_button.image = start_img
stop_button.image = stop_img
open_button.image = open_img

root.mainloop()
