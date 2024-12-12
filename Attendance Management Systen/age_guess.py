import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os

# Pre-trained model files (download and provide correct paths)
AGE_PROTO = "deploy_age.prototxt"
AGE_MODEL = "age_net.caffemodel"

# Load the age detection model
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

if not os.path.exists(AGE_PROTO) or not os.path.exists(AGE_MODEL):
    raise FileNotFoundError("Age detection model files not found. Please download and set correct paths.")

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# Face detection model
CASCADE_PATH = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(CASCADE_PATH)

# Predict Age Function
def predict_age(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return age

# Age Guessing Page
def guess_age_page():
    def start_guessing():
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                messagebox.showerror("Camera Error", "Unable to access the camera!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                age = predict_age(face_img)

                # Display age prediction
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Guess Age", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

    # GUI for Guess Age
    guess_root = tk.Tk()
    guess_root.title("Guess Age")
    guess_root.geometry("500x300")
    guess_root.configure(bg="#f0f0f0")

    tk.Label(guess_root, text="Guess Age Using Face", bg="#007BFF", fg="white", font=("Helvetica", 16, "bold"), anchor="w").pack(fill=tk.X, pady=20)

    start_button = tk.Button(guess_root, text="Start Guessing", bg="#28a745", fg="white", font=("Helvetica", 14), command=start_guessing)
    start_button.pack(pady=40)

    guess_root.mainloop()

# Main Page
def open_guess_age_page():
    root.destroy()
    guess_age_page()

root = tk.Tk()
root.title("Age Guessing System")
root.geometry("500x300")
root.configure(bg="#f0f0f0")

header_label = tk.Label(root, text="Age Guessing System", bg="#007BFF", fg="white", font=("Helvetica", 18, "bold"))
header_label.pack(fill=tk.X, pady=20)

guess_age_button = tk.Button(root, text="Guess Age", bg="#17a2b8", fg="white", font=("Helvetica", 14), command=open_guess_age_page)
guess_age_button.pack(pady=20)

footer_label = tk.Label(root, text="Developed by Dishant", bg="#f0f0f0", fg="#666666", font=("Helvetica", 10))
footer_label.pack(side=tk.BOTTOM, pady=10)

root.mainloop()
