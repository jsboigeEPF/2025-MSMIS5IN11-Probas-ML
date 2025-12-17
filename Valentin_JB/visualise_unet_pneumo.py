import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_pneumo_unet import UNet, PneumoDataset, CFG


MODEL_PATH = "best_unet_pneumo.pth"  # modèle entraîné sauvegardé
device = CFG.device


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Fichier modèle {MODEL_PATH} introuvable. "
            "Lance d'abord l'entraînement pour le créer."
        )

    model = UNet(in_ch=1, out_ch=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Modèle chargé depuis {MODEL_PATH} sur {device}")
    return model


def pick_random_sample():
    dataset = PneumoDataset(CFG.images_dir, CFG.masks_dir, img_size=CFG.img_size)
    idx = random.randint(0, len(dataset) - 1)
    img, mask, label = dataset[idx]  # img, mask: [1,H,W]
    print(f"[Sample] Index: {idx}, label (pneumothorax présent ?) = {label.item()}")
    return img, mask, label


def visualize_sample(model):
    img, gt_mask, label = pick_random_sample()

    # Prépare le batch
    img_batch = img.unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        logits = model(img_batch)
        probs = torch.sigmoid(logits)        # [1,1,H,W]
        pred_mask = (probs > 0.5).float()

    # On ramène tout en numpy pour affichage
    img_np = img.squeeze(0).cpu().numpy()          # [H,W]
    gt_np = gt_mask.squeeze(0).cpu().numpy()       # [H,W]
    pred_np = pred_mask.squeeze().cpu().numpy()    # [H,W]

    # Affichage
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="gray")
    axes[1].set_title("Masque GT")
    axes[1].axis("off")

    axes[2].imshow(img_np, cmap="gray")
    # on superpose le masque prédit en rouge translucide
    axes[2].imshow(pred_np, cmap="Reds", alpha=0.4)
    axes[2].set_title("Masque prédit")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    model = load_model()
    visualize_sample(model)


if __name__ == "__main__":
    main()
