import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras import models

MODEL_PATH = "tf-cnn-model.h5"
model = None

def load_model():
    global model
    model = models.load_model(MODEL_PATH, compile=False)

def predict_digit(image_path):
    image = cv2.imread(image_path, 0)
    image1 = cv2.resize(image, (28, 28))
    image2 = image1.reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]

def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select a digit image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # Show image in UI
    img = Image.open(file_path).convert("L").resize((200, 200), Image.NEAREST)
    img_color = img.convert("RGB")
    photo = ImageTk.PhotoImage(img_color)
    image_label.config(image=photo)
    image_label.image = photo
    image_label.config(text="")

    # Predict
    digit = predict_digit(file_path)
    result_label.config(text=str(digit))
    result_text.config(text=f"The model predicts this is the digit:")
    file_label.config(text=file_path.split("/")[-1])

# --------------- UI Setup ---------------
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("420x560")
root.resizable(False, False)
root.configure(bg="#1e1e2e")

title_font  = font.Font(family="Helvetica", size=18, weight="bold")
label_font  = font.Font(family="Helvetica", size=12)
result_font = font.Font(family="Helvetica", size=72, weight="bold")
small_font  = font.Font(family="Helvetica", size=10)

# Title
tk.Label(root, text="Digit Recognizer", font=title_font,
         bg="#1e1e2e", fg="#cdd6f4").pack(pady=(20, 4))
tk.Label(root, text="Handwritten digit recognition using CNN",
         font=small_font, bg="#1e1e2e", fg="#6c7086").pack()

# Image display box
image_frame = tk.Frame(root, bg="#313244", width=210, height=210,
                        relief="flat", bd=0)
image_frame.pack(pady=20)
image_frame.pack_propagate(False)

image_label = tk.Label(image_frame, bg="#313244", fg="#6c7086",
                        text="No image selected", font=small_font)
image_label.pack(expand=True)

# File name
file_label = tk.Label(root, text="", font=small_font,
                       bg="#1e1e2e", fg="#6c7086")
file_label.pack()

# Browse button
browse_btn = tk.Button(root, text="  Browse Image  ", font=label_font,
                        bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
                        activeforeground="#1e1e2e", relief="flat", cursor="hand2",
                        command=browse_image, padx=10, pady=6)
browse_btn.pack(pady=12)

# Result area
result_text = tk.Label(root, text="", font=small_font,
                        bg="#1e1e2e", fg="#a6adc8")
result_text.pack()

result_label = tk.Label(root, text="", font=result_font,
                         bg="#1e1e2e", fg="#a6e3a1")
result_label.pack()

# Load model on startup
root.after(100, load_model)
root.mainloop()
