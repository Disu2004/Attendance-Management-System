import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
from PIL import Image

# Paths
IMAGES_PATH = "StudentImages"
MODEL_PATH = "TrainingImageLabel/Trainer.yml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Ensure directories exist
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
if not os.path.exists("TrainingImageLabel"):
    os.makedirs("TrainingImageLabel")

# Initialize Face Recognizer and Cascade Classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(CASCADE_PATH)

# Register Page
def register_page():
    def take_images_and_train():
        roll_no = enrollment_entry.get()
        name = name_entry.get()

        if not roll_no or not name:
            messagebox.showerror("Input Error", "Please enter both Enrollment and Name!")
            return

        # Capture new images
        camera = cv2.VideoCapture(0)
        count = 0
        captured_faces = []

        while True:
            ret, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                captured_faces.append(face_img)
                count += 1
                face_path = os.path.join(IMAGES_PATH, f"{name}.{roll_no}.{count}.jpg")
                cv2.imwrite(face_path, face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Capturing Images", frame)

            if cv2.waitKey(1) & 0xFF == ord("q") or count >= 50:
                break

        camera.release()
        cv2.destroyAllWindows()

        if count > 0:
            # Check if captured faces match any existing images
            existing_image_paths = [os.path.join(IMAGES_PATH, f) for f in os.listdir(IMAGES_PATH) if f.endswith(".jpg")]
            for existing_image_path in existing_image_paths:
                existing_img = Image.open(existing_image_path).convert("L")
                existing_img_np = np.array(existing_img, "uint8")

                for face in captured_faces:
                    id_, conf = recognizer.predict(face)
                    if conf < 50:  # Confidence threshold for duplicate face
                        messagebox.showerror("Duplicate Error", "Face already exists in the system!")
                        return

            train_images()
            messagebox.showinfo("Success", f"Registration successful for {name}!")
        else:
            messagebox.showerror("Error", "No images were captured!")

    def train_images():
        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
            face_samples = []
            ids = []

            for image_path in image_paths:
                pil_image = Image.open(image_path).convert("L")
                image_np = np.array(pil_image, "uint8")
                id_ = int(os.path.split(image_path)[-1].split(".")[1])  # Extract ID
                faces = detector.detectMultiScale(image_np)

                for (x, y, w, h) in faces:
                    face_samples.append(image_np[y:y+h, x:x+w])
                    ids.append(id_)

            return face_samples, ids

        faces, ids = get_images_and_labels(IMAGES_PATH)

        if not faces:
            messagebox.showerror("Training Error", "No images found for training!")
            return

        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_PATH)

    # Register GUI
    register_root = tk.Tk()
    register_root.title("Register New Student")
    register_root.geometry("500x350")
    register_root.configure(bg="#f0f0f0")

    tk.Label(register_root, text="Register New Student", bg="#007BFF", fg="white", font=("Helvetica", 16, "bold"), anchor="w").pack(fill=tk.X)

    tk.Label(register_root, text="Enter Enrollment:", bg="#f0f0f0", fg="#333333", font=("Helvetica", 12)).pack(pady=10)
    enrollment_entry = tk.Entry(register_root, width=40, font=("Helvetica", 12))
    enrollment_entry.pack(pady=5)

    tk.Label(register_root, text="Enter Name:", bg="#f0f0f0", fg="#333333", font=("Helvetica", 12)).pack(pady=10)
    name_entry = tk.Entry(register_root, width=40, font=("Helvetica", 12))
    name_entry.pack(pady=5)

    register_button = tk.Button(register_root, text="Register", bg="#28a745", fg="white", font=("Helvetica", 14), command=take_images_and_train)
    register_button.pack(pady=20)

    register_root.mainloop()

# Mark Attendance Page
def mark_attendance():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Model Missing", "Please train the model first!")
        return

    recognizer.read(MODEL_PATH)
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            confidence = round(100 - conf, 2)

            if conf < 50:  # Confidence threshold
                name = f"Roll {id_} ({confidence}%)"
                color = (0, 255, 0)
                messagebox.showinfo("Attendance", f"Attendance marked successfully for Roll {id_}!")
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

# Main Page
def open_mark_attendance_page():
    root.destroy()
    mark_attendance()

def open_register_page():
    root.destroy()
    register_page()

root = tk.Tk()
root.title("Attendance Management System")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

header_label = tk.Label(root, text="Attendance Management System", bg="#007BFF", fg="white", font=("Helvetica", 18, "bold"))
header_label.pack(fill=tk.X, pady=20)

mark_attendance_button = tk.Button(root, text="Mark Attendance", bg="#17a2b8", fg="white", font=("Helvetica", 14), command=open_mark_attendance_page)
mark_attendance_button.pack(pady=20)

register_button = tk.Button(root, text="Register New Student", bg="#28a745", fg="white", font=("Helvetica", 14), command=open_register_page)
register_button.pack(pady=20)

footer_label = tk.Label(root, text="Developed by Dishant", bg="#f0f0f0", fg="#666666", font=("Helvetica", 10))
footer_label.pack(side=tk.BOTTOM, pady=10)

root.mainloop()