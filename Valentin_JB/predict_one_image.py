import os
import torch
import numpy as np
from PIL import Image
from train_pneumo_unet import UNet, CFG  # on réutilise le modèle et la config

# 1) Chemins
MODEL_PATH = "best_unet_pneumo.pth"
IMAGE_PATH = r"C:\Users\valen\OneDrive\Documents\EPF\COURS\5A\IA\2025-MSMIS5IN11-Probas-ML\radio_test_2.jpg"

device = CFG.device

# 2) Charger le modèle
model = UNet(in_ch=1, out_ch=1).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 3) Préparer l'image
img = Image.open(IMAGE_PATH).convert("L")
img = img.resize((CFG.img_size, CFG.img_size))
img_np = np.array(img, dtype=np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

# 4) Prédiction
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)           # [1,1,H,W]
    mask = (probs > 0.5).float()            # binaire
    has_pneumo = (mask.sum() > 0).item()    # 0 ou 1

print("Pneumothorax détecté ?" , bool(has_pneumo))
print("Masque - valeur max:", mask.max().item())
