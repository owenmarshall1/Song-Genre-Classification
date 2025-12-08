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



from images_model import Deit   
from images_dataset import ImageGenreDataset

def predict(model,image,class_names):
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    image = transform(image) 

    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        predict = logits.argmax(dim=1).item()
    predicted_class  = class_names[predict]
    return predicted_class

def audio_to_image(audio_path):
    y, sr = librosa.load(audio_path)

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

########
device = "cuda" if torch.cuda.is_available() else "cpu"

data = ImageGenreDataset("data/images_original")
class_names = data.classes
num_classes = len(class_names)

model = Deit(num_classes , )
model.load_state_dict(torch.load("trained_model.pth"))
model.to(device)


predictions = []
for i in range(3):
    img = audio_to_image("songs/Frank_Sinatra_–_Cheek_To_Cheek_T.mp3")
    pred = predict(model, img, class_names)
    predictions.append(pred)
prediction = Counter(predictions).most_common(1)[0][0]
print(prediction)