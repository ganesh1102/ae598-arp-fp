"""Extract ResNet-18 features from RGB images stored in HDF5."""
import argparse
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


def get_encoder(device):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()
    resnet.to(device)
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract(h5_path, output_path, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    encoder = get_encoder(device)
    transform = get_transform()

    with h5py.File(h5_path, 'r') as f:
        rgb = f['images/rgb']
        N = rgb.shape[0]
        print(f"Extracting features for {N} images")

        features = np.zeros((N, 512), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = rgb[start:end]  # (B, H, W, 3) uint8

            tensors = torch.stack([transform(img) for img in batch]).to(device)

            with torch.no_grad():
                feats = encoder(tensors).cpu().numpy()
            features[start:end] = feats

            if (start // batch_size) % 10 == 0:
                print(f"  {end}/{N}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, features)
    print(f"Saved {output_path}, shape={features.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='legged-loco/data/obstacle_dataset.h5')
    parser.add_argument('--out', default='data/trigger_features_real.npy')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    extract(args.h5, args.out, args.batch_size)