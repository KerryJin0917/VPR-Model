#!/usr/bin/env python3
"""
ResNet50 Baseline for Visual Place Recognition

Generates prediction CSV files for evaluation. Uses pretrained ResNet50
to encode images, then ranks database items for each query.

Usage:
    python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --dataset_name dataset_a --output predictions/dataset_a.csv
    python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --dataset_name dataset_b --output predictions/dataset_b.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root, image_paths, image_size=(224, 224)):
        self.root = root
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


class ResNetEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.to(device).eval()

    @torch.inference_mode()
    def encode(self, images):
        images = images.to(self.device)
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        return F.normalize(self.backbone(images), p=2, dim=1)


def encode_images(model, dataset, batch_size=64, num_workers=4):
    """Encode all images and return embeddings in order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    emb_list, idx_list = [], []
    for images, indices in tqdm(loader, desc="Encoding"):
        emb_list.append(model.encode(images).cpu().numpy())
        idx_list.append(indices.numpy())
    embeddings = np.vstack(emb_list)
    indices = np.concatenate(idx_list)
    return embeddings[np.argsort(indices)]


def load_dataset_a(root):
    """Load dataset_a: sorted database and query image paths."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))
    db_df = df[df["split"] == "database"].sort_values("image_path")
    q_df = df[df["split"] == "queries"].sort_values("image_path")
    return db_df["image_path"].tolist(), q_df["image_path"].tolist()


def load_dataset_b(root):
    """Load dataset_b: database and query paths from test.parquet."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))
    db_df = df[df["role"] == "database"]
    q_df = df[df["role"] == "queries"]
    return db_df["image_path"].tolist(), q_df["image_path"].tolist()


def main():
    parser = argparse.ArgumentParser(description="ResNet50 Baseline - Generate VPR Predictions")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["dataset_a", "dataset_b"])
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20, help="Number of ranked results per query")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Load paths
    if args.dataset_name == "dataset_a":
        db_paths, query_paths = load_dataset_a(args.dataset_root)
    else:
        db_paths, query_paths = load_dataset_b(args.dataset_root)

    print(f"Database: {len(db_paths)}, Queries: {len(query_paths)}")

    model = ResNetEncoder(args.device)

    # Encode
    db_dataset = ImageDataset(args.dataset_root, db_paths)
    query_dataset = ImageDataset(args.dataset_root, query_paths)

    db_emb = encode_images(model, db_dataset, args.batch_size, args.num_workers)
    query_emb = encode_images(model, query_dataset, args.batch_size, args.num_workers)

    # Compute rankings
    print("Computing rankings...")
    similarity = np.matmul(query_emb, db_emb.T)
    rankings = np.argsort(-similarity, axis=1)[:, :args.top_k]

    # Save predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_idx in range(len(query_paths)):
        ranked_str = ",".join(str(x) for x in rankings[q_idx])
        rows.append({"query_index": q_idx, "ranked_database_indices": ranked_str})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
