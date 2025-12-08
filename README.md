# **Song Genre Classification**  
*(CSCI 4050U — Final Machine Learning Project)*

# Group Members
Owen Marshall - 100916209

Seyed Arshia Hashemianzadeh - 100870762

## **1. Project Overview**
This project explores the problem of **song genre classification** by converting raw audio into **Mel-spectrogram images** and applying **Vision Transformer (ViT)-based architectures** for supervised multi-class classification. the main challenge was training a classifier with limited data and high accuracy.

Our primary objective is to use a strong and efficient **custom-built Vision Transformer (ViT)** while solving the problems of having small dataset by utilizing **pretrained DeiT (Data-efficient Image Transformer)** whose design and features help reduce overfitting.:

- A transformer trained from scratch on our dataset  
- A transformer benefiting from ImageNet pretraining  

The final system includes:  
- A Python-based training pipeline  
- Two distinct Transformer architectures  
- A Tkinter-based GUI for song-genre prediction  
- A complete deployment setup for local use
---

## **2. Dataset**
The dataset we obtained is the GTZAN dataset, one of the most-used public dataset for evaluation in machine listening research for music genre recognition. it consists of a collection of 10 genres with 100 audio files each, all having a length of 30 seconds. 

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

## **3. Model Architectures**

### **A. Custom Vision Transformer (ViT)**
Key components:

- Patch embedding via linear projection  
- Learnable classification token  
- Transformer encoder with multi-head self-attention  
- MLP classification head  

This model is trained **entirely from scratch**, learning directly from the dataset. this model however, requires large amount of data and since the dataset used only consisted of 1000 songs it didn't have enough data to more accurately train.

---

### **B. Pretrained DeiT (Data-Efficient Image Transformer)**
Loaded from the **timm** library.

- Pretrained on **ImageNet**  
- Fine-tuned for our dataset  
- Expected to outperform the custom ViT due to transfer learning  
- Highlights differences between pretrained vs. non-pretrained transformers  

---

## **4. Training Approach**
Training includes:

- AdamW optimizer  
- Cosine learning-rate scheduler  
- 5-epoch warmup  
- Automatic saving of models:  
  - `trained_model.pth` (DeiT)  
  - `ViT_model.pth` (Custom ViT)

### **Example Training Commands**
```bash
python train.py --mode deit --epochs 50
python train.py --mode vit --epochs 50
```

Evaluation uses final test accuracy and test loss metrics.

---

## **5. GUI Deployment (Local Application)**

The **Tkinter GUI** allows users to upload an audio file and receive:

- A generated spectrogram  
- Genre prediction  
- Confidence percentage  
- Ability to switch between DeiT and ViT models  

### **How to Run the GUI**

#### **Using executables:**

**Windows:**  
Run:  
```
run_windows.bat
```

**Mac/Linux:**  
Run:  
```
run_shell.sh
```

Both should automatically install dependencies.  
If they do not, install manually:

#### **In Console**
```bash
pip install -r requirements.txt
python gui_predict.py
```

---

## **6. Conclusion**
This project demonstrates a complete machine learning pipeline:

- successfully designed, trained, and deployed an AI system
- Data preprocessing and augmentation  
- Two transformer-based neural network architectures  
- Comparative evaluation of pretrained vs. non-pretrained models  

