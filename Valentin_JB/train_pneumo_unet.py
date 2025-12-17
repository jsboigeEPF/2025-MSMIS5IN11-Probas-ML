import os
import glob
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# CONFIG
# ============================================================

# TODO: MODIFIE CE CHEMIN AVEC LE CHEMIN DE TON DOSSIER LOCAL
# Exemple : r"C:\Users\Valentin\Documents\pneumothorax"
DATA_ROOT = r"C:\Users\valen\Downloads\Pneumothorax"

class CFG:
    data_root = DATA_ROOT
    images_dir = os.path.join(data_root, "png_images")
    masks_dir = os.path.join(data_root, "png_masks")

    img_size = 128
    batch_size = 8
    num_workers = 0         # mets >0 si tu es sous Linux, 0 évite les soucis sous Windows
    lr = 1e-4
    num_epochs = 2
    val_ratio = 0.15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

# ============================================================
# DATASET
# ============================================================

class PneumoDataset(Dataset):
    """
    Dataset pour images + masques de pneumothorax.

    On suppose que :
      - les images sont dans `images_dir`
      - les masques sont dans `masks_dir`
      - soit le masque a exactement le même nom que l'image,
      - soit le masque a le suffixe `_mask` avant l'extension.
    """

    def __init__(self, images_dir: str, masks_dir: str, img_size: int = 256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image .png trouvée dans {images_dir}")

        print(f"[Dataset] {len(self.image_paths)} images trouvées")

    def _mask_path_from_image_path(self, img_path: str) -> str:
        """
        Essaie d'abord 'nom.png' dans png_masks.
        Si inexistant, essaie 'nom_mask.png'.
        Adapte ici si besoin selon tes fichiers.
        """
        basename = os.path.basename(img_path)

        # 1) même nom
        mask_path_same = os.path.join(self.masks_dir, basename)
        if os.path.exists(mask_path_same):
            return mask_path_same

        # 2) nom_mask.png
        name, ext = os.path.splitext(basename)
        mask_basename = f"{name}_mask{ext}"
        mask_path_mask = os.path.join(self.masks_dir, mask_basename)
        if os.path.exists(mask_path_mask):
            return mask_path_mask

        # Si rien trouvé -> erreur
        raise FileNotFoundError(
            f"Masque introuvable pour {img_path}.\n"
            f"Essayé : {mask_path_same} et {mask_path_mask}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> Image.Image:
        # convertit tout en niveaux de gris
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size))
        return img

    def _load_mask(self, path: str) -> Image.Image:
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        return mask

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self._mask_path_from_image_path(img_path)

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        img = np.array(img, dtype=np.float32) / 255.0      # [H, W]
        mask = np.array(mask, dtype=np.float32) / 255.0    # [H, W]

        # Label binaire : 1 s'il y a au moins un pixel positif dans le masque
        label = float(mask.max() > 0.5)

        # [C, H, W]
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return img, mask, label

# ============================================================
# MODELE U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()

        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.down1(x)
        p1 = self.maxpool(c1)

        c2 = self.down2(p1)
        p2 = self.maxpool(c2)

        c3 = self.down3(p2)
        p3 = self.maxpool(c3)

        c4 = self.down4(p3)
        p4 = self.maxpool(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        c5 = self.dec4(u4)

        u3 = self.up3(c5)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.dec3(u3)

        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.dec2(u2)

        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.dec1(u1)

        out = self.out_conv(c8)
        return out  # logits, on mettra un sigmoid après


# ============================================================
# LOSS & METRICS
# ============================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred et target sont des probas [B,1,H,W]
    """
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = num / den
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        d = dice_loss(probs, targets)
        return bce + d


def segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    preds, targets: [B,1,H,W] (après sigmoid).
    Retourne dice, IoU moyens.
    """
    preds_bin = (preds > threshold).float()

    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    dice = (2 * intersection + eps) / (
        preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    )
    iou = (intersection + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


def detection_metrics(
    preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    preds: masques de probas [B,1,H,W]
    labels: [B] (0 ou 1)
    On convertit le masque en label binaire : au moins un pixel > threshold.
    """
    with torch.no_grad():
        preds_bin = (preds > threshold).float()
        pred_labels = (preds_bin.sum(dim=(1, 2, 3)) > 0).float()

        tp = ((pred_labels == 1) & (labels == 1)).sum().item()
        tn = ((pred_labels == 0) & (labels == 0)).sum().item()
        fp = ((pred_labels == 1) & (labels == 0)).sum().item()
        fn = ((pred_labels == 0) & (labels == 1)).sum().item()

        total = tp + tn + fp + fn + 1e-6
        acc = (tp + tn) / total
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)

    return acc, prec, rec, f1


# ============================================================
# TRAIN / VAL
# ============================================================

def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    dataset = PneumoDataset(CFG.images_dir, CFG.masks_dir, img_size=CFG.img_size)

    # ⚠️ IMPORTANT : on prend seulement une partie du dataset pour aller plus vite
    max_samples = 2000  # par exemple 2000 images au lieu de 12047
    if len(dataset) > max_samples:
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"[Data] Sous-échantillonnage à {max_samples} images")

    n_total = len(dataset)
    n_val = int(n_total * CFG.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    print(f"[Data] Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int
) -> float:
    model.train()
    running_loss = 0.0

    max_batches = 300  # coupe l'epoch à 300 batches max

    for i, (imgs, masks, labels) in enumerate(train_loader):
        imgs = imgs.to(CFG.device)
        masks = masks.to(CFG.device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

        if (i + 1) >= max_batches:
            print(f"⏹️  Epoch {epoch} coupée à {max_batches} batches pour aller plus vite")
            break

    # on normalise par le nombre de batches effectivement utilisés
    used_batches = min(len(train_loader), max_batches)
    epoch_loss = running_loss / max(used_batches, 1)
    return epoch_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    epoch: int
) -> float:
    model.eval()
    val_loss = 0.0
    dices: List[float] = []
    ious: List[float] = []
    accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []

    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs = imgs.to(CFG.device)
            masks = masks.to(CFG.device)
            labels = labels.to(CFG.device)

            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            d, iou = segmentation_metrics(probs, masks)
            dices.append(d)
            ious.append(iou)

            acc, prec, rec, f1 = detection_metrics(probs, labels)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

    val_loss /= max(len(val_loader), 1)

    print(f"\n==== Validation Epoch {epoch} ====")
    print(f"Val loss       : {val_loss:.4f}")
    print(f"Mean Dice      : {np.mean(dices):.4f}")
    print(f"Mean IoU       : {np.mean(ious):.4f}")
    print(
        "Detection - "
        f"Acc: {np.mean(accs):.4f}, "
        f"Prec: {np.mean(precs):.4f}, "
        f"Rec: {np.mean(recs):.4f}, "
        f"F1: {np.mean(f1s):.4f}"
    )
    print("=================================\n")

    return val_loss


# ============================================================
# MAIN
# ============================================================

def main():
    # Vérifier que les dossiers existent
    if not os.path.isdir(CFG.images_dir):
        raise RuntimeError(f"Dossier images introuvable : {CFG.images_dir}")
    if not os.path.isdir(CFG.masks_dir):
        raise RuntimeError(f"Dossier masques introuvable : {CFG.masks_dir}")

    print(f"Device utilisé : {CFG.device}")

    train_loader, val_loader = get_dataloaders()

    model = UNet(in_ch=1, out_ch=1).to(CFG.device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_val_loss = float("inf")

    for epoch in range(1, CFG.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        print(f"Epoch {epoch} - Train loss : {train_loss:.4f}")

        val_loss = validate(model, val_loader, criterion, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_pneumo.pth")
            print(
                f"✅ Nouveau meilleur modèle sauvegardé "
                f"(epoch {epoch}, val_loss={val_loss:.4f})"
            )


if __name__ == "__main__":
    main()

