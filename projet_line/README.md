# DCGAN Medical Demo (MNIST proxy)

This repository contains a simple DCGAN implemented in PyTorch intended as a demo
for generating synthetic images as a proxy for medical data augmentation.

Files:
- `dcgan_medical_demo.py`: main script implementing Generator and Discriminator,
  training loop, fixed noise visualizations and image saving to `progress_images/`.
- `requirements.txt`: minimal Python dependencies.

Quick start (PowerShell):

```powershell
python -m pip install -r requirements.txt
python dcgan_medical_demo.py --dataset MNIST --epochs 10 --batch_size 128
```

After running, generated grids will be saved in `progress_images/` as `epoch_001.png`,
`epoch_002.png`, etc. These images are produced from a fixed latent vector so you can
visually track how the generator improves over time.

Adapting to medical images (notes):
- Replace the dataset loading part with a custom `torch.utils.data.Dataset` that
  reads DICOM/PNG/JPEG medical images and applies appropriate transforms (windowing,
  normalization). Ensure images are resized/cropped to the `--image_size` used.
- For multi-channel or higher-resolution images (X-Ray, MRI), increase `nc` and
  the network capacity (`ngf`, `ndf`) or use progressive training.
- Clinical validation is required before using generated images for model training.

Security & ethics:
- Synthetic medical images must not be used as-is in clinical decision making.
- Always get domain expert validation and consider patient privacy and data governance.
