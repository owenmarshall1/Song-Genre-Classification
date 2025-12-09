import torch
from PIL import Image
import torch
from torchvision import transforms

from collections import Counter
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import os



from images_model import Deit   
from images_dataset import ImageGenreDataset

###predicting function for spectrogram image
def predict(model,image,class_names):

    # apply transfers that were applied to data 
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.3, 0.3, 0.3]
    )])
    image = transform(image) 

    #forward pass
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        predict = logits.argmax(dim=1).item()
    predicted_class  = class_names[predict]
    return predicted_class

###function for turning audio to first 30 second spectrogram 
def audio_to_image(audio_path):

    # load and normalize audio (first 30 seconds)
    y, sr = librosa.load(audio_path, duration =30)
    y = librosa.util.normalize(y)

    #compute and plot spectogram using librosa 
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(3, 3), dpi=80)
    plt.axis("off")
    librosa.display.specshow(S_dB, sr=sr, cmap="viridis")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")

    return img

########main##########
device = "cuda" if torch.cuda.is_available() else "cpu"

data = ImageGenreDataset("data/images_original")
class_names = data.classes
num_classes = len(class_names)

model = Deit(num_classes)
model.load_state_dict(torch.load("trained_model.pth"))
model.to(device)

###for manual testing

# for song in os.listdir("songs/"): 
#     path = os.path.join("songs/", song)
#     predictions = []
#     for i in range(20):
#         img = audio_to_image(path)
#         pred = predict(model, img, class_names)
#         predictions.append(pred)
#     prediction = Counter(predictions).most_common(1)[0][0]
#     print(f"{song} → {prediction}")