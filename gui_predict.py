import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os

from predict import audio_to_image, predict
from images_model import ImageViT, Deit
from images_dataset import ImageGenreDataset
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

data = ImageGenreDataset("data/images_original")
class_names = data.classes
num_classes = len(class_names)

current_model = "deit"

selected_model = None

current_filepath = None

def load_model(name):
    global model, current_model, selected_model
    prev = current_model
    current_model = name
    if name == "deit":
        model = Deit(num_classes)
        weight_path = "trained_model.pth"
    else:
        model = ImageViT(num_classes)
        weight_path = "ViT_model.pth"

    if not os.path.exists(weight_path):
        messagebox.showerror("Error", f"Model weights file '{weight_path}' not found.")
        if selected_model is not None:
            selected_model.set(prev)
        current_model = prev
        return

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model weights:\n{e}")
        if selected_model is not None:
            selected_model.set(prev)
        current_model = prev
        return

    update_model_buttons()

def update_model_buttons():
    sel = selected_model.get() if selected_model is not None else current_model
    highlight = "#4a90e2"
    normal_bg = "#2b2b2b"
    if sel == "deit":
        btn_deit.config(bg=highlight, fg="black", relief="sunken")
        btn_vit.config(bg=normal_bg, fg="white", relief="raised")
    else:
        btn_vit.config(bg=highlight, fg="black", relief="sunken")
        btn_deit.config(bg=normal_bg, fg="white", relief="raised")

def select_file():
    global current_filepath
    filepath = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.mp3 *.wav")]
    )
    if not filepath:
        return

    current_filepath = filepath
    lbl_file.config(text=os.path.basename(filepath))
    predict_song(filepath)

def get_filepath():
    return current_filepath

def predict_song(filepath):
    try:
        image = audio_to_image(filepath)
        image_resized = image.resize((200,200))
        tk_image = ImageTk.PhotoImage(image_resized)
        lbl_image.config(image=tk_image)
        lbl_image.image = tk_image

        tensor_image = transforms_image(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor_image)
            probs = torch.softmax(logits, dim=1)[0]

        predicted_index = torch.argmax(probs).item()
        predicted_class = class_names[predicted_index]
        confidence = probs[predicted_index].item() * 100

        lbl_result.config(
            text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",
            fg="white"
        )


    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

import torchvision.transforms as transforms
transforms_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

############### GUI Setup ###############
root = tk.Tk()
root.title("Song Genre Classifier")
root.geometry("500x600")
root.configure(bg="#1e1e1e")

lbl_title = tk.Label(root, text="Song Genre Classifier", font=("Arial", 20, "bold"), bg="#1e1e1e", fg="white")
lbl_title.pack(pady=10)

frame_models = tk.Frame(root, bg="#1e1e1e")
frame_models.pack(pady=10)

selected_model = tk.StringVar(value=current_model)

btn_deit = tk.Radiobutton(
    frame_models,
    text="Deit (pretrained)",
    variable=selected_model,
    value="deit",
    indicatoron=0,
    width=18,
    font=("Arial", 11, "bold"),
    command=lambda: (load_model("deit"), predict_song(get_filepath()) if get_filepath() else None),
    bg="#2b2b2b",
    fg="white",
    bd=0,
    activebackground="#357ab8",
    padx=8,
    pady=6
)
btn_deit.grid(row=0, column=0, padx=8)

btn_vit = tk.Radiobutton(
    frame_models,
    text="ViT (custom)",
    variable=selected_model,
    value="vit",
    indicatoron=0,
    width=18,
    font=("Arial", 11, "bold"),
    command=lambda: (load_model("vit"), predict_song(get_filepath()) if get_filepath() else None),
    bg="#2b2b2b",
    fg="white",
    bd=0,
    activebackground="#357ab8",
    padx=8,
    pady=6
)
btn_vit.grid(row=0, column=1, padx=8)

selected_model = selected_model

load_model("deit")
update_model_buttons()

btn_upload = tk.Button(root, text="Upload Song", command=select_file, font=("Arial", 14))
btn_upload.pack(pady=10)

lbl_file = tk.Label(root, text="No file selected", bg="#1e1e1e", fg="white")
lbl_file.pack()

lbl_image = tk.Label(root, bg="#1e1e1e")
lbl_image.pack(pady=10)

lbl_result = tk.Label(root, text="", font=("Arial", 16), bg="#1e1e1e", fg="white")
lbl_result.pack(pady=20)

root.mainloop()
