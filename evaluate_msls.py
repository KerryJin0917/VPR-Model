#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from your_model_file import TrainableModel  # adjust import


# ---------------------------
# Dataset loader (MSLS style)
# ---------------------------
class MSLSDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size=224):
        self.image_paths = image_paths

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), idx


# ---------------------------
# Encoding
# ---------------------------
@torch.no_grad()
def encode(model, loader, device):
    embs = []
    idxs = []

    for imgs, idx in tqdm(loader, desc="Encoding"):
        imgs = imgs.to(device)
        feat = model.encode(imgs)

        embs.append(feat.cpu())
        idxs.append(idx)

    embs = torch.cat(embs).numpy()
    idxs = torch.cat(idxs).numpy()

    return embs[np.argsort(idxs)]


# ---------------------------
# Recall@K
# ---------------------------
def recall_at_k(sim, q_labels, db_labels, ks=(1, 5, 10)):
    rankings = np.argsort(-sim, axis=1)

    results = {}
    for k in ks:
        correct = 0

        for i, ql in enumerate(q_labels):
            topk = rankings[i][:k]
            if any(db_labels[j] == ql for j in topk):
                correct += 1

        results[k] = correct / len(q_labels)

    return results


# ---------------------------
# Main
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", args.checkpoint)

    # Load MSLS metadata
    df = pd.read_csv(args.csv)

    db_df = df[df["split"] == "database"]
    q_df = df[df["split"] == "query"]

    db_paths = db_df["image_path"].tolist()
    q_paths = q_df["image_path"].tolist()

    db_labels = db_df["place_id"].values
    q_labels = q_df["place_id"].values

    print(f"DB: {len(db_paths)}, Queries: {len(q_paths)}")

    # DataLoaders
    db_loader = torch.utils.data.DataLoader(
        MSLSDataset(db_paths, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    q_loader = torch.utils.data.DataLoader(
        MSLSDataset(q_paths, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Encode
    db_emb = encode(model, db_loader, device)
    q_emb = encode(model, q_loader, device)

    # Similarity
    sim = q_emb @ db_emb.T

    # Evaluate
    results = recall_at_k(sim, q_labels, db_labels)

    print("\n=== MSLS RESULTS ===")
    for k, v in results.items():
        print(f"Recall@{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=512)
    args = parser.parse_args()

    main(args)