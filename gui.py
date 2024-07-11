import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2

# Load the pre-trained model
def load_model_from_file(model_file):
    model = load_model(model_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the GUI window
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Signal Predictor')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the model
model = load_model_from_file("C:/Users/harsh/Desktop/Task_2_2/traffic_signal_model.h5")

CLASSES_LIST = ["car", "people", "other_vehicles"]
COLOR_SWAP = {'red': 'blue', 'blue': 'red'}

def predict(file_path):
    global label1

    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, (128, 128))
    image_resized = image_resized / 255.0
    prediction = model.predict(np.expand_dims(image_resized, axis=0))
    class_pred = CLASSES_LIST[np.argmax(prediction)]

    label1.configure(foreground="#011638", text=class_pred)

    if class_pred == "car":
        detect_cars(file_path)
    elif class_pred == "people":
        detect_people(file_path)
    else:
        detect_other_vehicles(file_path)

def detect_cars(file_path):
    image = cv2.imread(file_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Detect red cars
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    red_cars = cv2.bitwise_and(image, image, mask=red_mask)

    # Detect blue cars
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    blue_cars = cv2.bitwise_and(image, image, mask=blue_mask)

    # Swap colors
    image[np.where(red_mask)] = [255, 0, 0]
    image[np.where(blue_mask)] = [0, 0, 255]

    cv2.imshow("Red Cars Swapped to Blue and Blue Cars Swapped to Red", image)
    cv2.waitKey(0)

def detect_people(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facec = cv2.CascadeClassifier("C:/Users/harsh/Desktop/Task_2_2/haarcasacade_frontalface_default.xml")
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    
    male_count = 0
    female_count = 0

    for (x, y, w, h) in faces:
        face = gray_image[y:y+h, x:x+w]
        roi = cv2.resize(face, (128, 128))
        pred = model.predict(np.expand_dims(roi, axis=0))
        gender = np.argmax(pred)
        if gender == 0:
            male_count += 1
        else:
            female_count += 1

    label1.configure(foreground="#011638", text=f"Males: {male_count}, Females: {female_count}")

def detect_other_vehicles(file_path):
    # Placeholder for actual detection of other vehicles
    label1.configure(foreground="#011638", text="Detected other vehicles")

def show_predict_button(file_path):
    predict_b = Button(top, text="Predict", command=lambda: predict(file_path), padx=10, pady=5)
    predict_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    predict_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_predict_button(file_path)
    except Exception as e:
        print(e)
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Traffic Signal Predictor', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
