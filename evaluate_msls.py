#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# IMPORT YOUR MODEL
# ============================================================
from LearnerPR import TrainableModel  # <-- change this


# ============================================================
# DATASET (folder-based, no CSV needed)
# ============================================================
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_size=224):
        self.paths = paths

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


# ============================================================
# ENCODING
# ============================================================
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


# ============================================================
# RECALL@K
# ============================================================
def recall_at_k(sim, q_labels, db_labels, ks=(1, 5, 10)):
    ranks = np.argsort(-sim, axis=1)

    results = {}

    for k in ks:
        correct = 0

        for i, ql in enumerate(q_labels):
            topk = ranks[i][:k]
            if any(db_labels[j] == ql for j in topk):
                correct += 1

        results[k] = correct / len(q_labels)

    return results


# ============================================================
# CITY EVALUATION
# ============================================================
def evaluate_city(model, city_path, image_size, batch_size, device):
    db_path = os.path.join(city_path, "database")
    q_path = os.path.join(city_path, "query")

    db_imgs = sorted([
        os.path.join(db_path, f)
        for f in os.listdir(db_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    q_imgs = sorted([
        os.path.join(q_path, f)
        for f in os.listdir(q_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    # --------------------------------------------------------
    # MSLS-style weak labels (filename-based grouping)
    # --------------------------------------------------------
    db_labels = [os.path.basename(f).split("_")[0] for f in db_imgs]
    q_labels = [os.path.basename(f).split("_")[0] for f in q_imgs]

    db_loader = torch.utils.data.DataLoader(
        ImageDataset(db_imgs, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    q_loader = torch.utils.data.DataLoader(
        ImageDataset(q_imgs, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    db_emb = encode(model, db_loader, device)
    q_emb = encode(model, q_loader, device)

    sim = q_emb @ db_emb.T

    return recall_at_k(sim, q_labels, db_labels)


# ============================================================
# FULL MSLS EVALUATION
# ============================================================
def evaluate_all_cities(model, root, image_size, batch_size, device):
    cities = [
        c for c in os.listdir(root)
        if os.path.isdir(os.path.join(root, c))
    ]

    results = {}

    for city in cities:
        city_path = os.path.join(root, city)

        if not os.path.exists(os.path.join(city_path, "database")):
            continue

        print(f"\n====================")
        print(f"City: {city}")
        print(f"====================")

        res = evaluate_city(
            model,
            city_path,
            image_size,
            batch_size,
            device
        )

        results[city] = res

        print(f"{city} results: {res}")

    return results


# ============================================================
# AVERAGE RESULTS
# ============================================================
def average_results(results):
    ks = [1, 5, 10]

    avg = {k: 0 for k in ks}
    n = len(results)

    for city, res in results.items():
        for k in ks:
            avg[k] += res[k]

    for k in ks:
        avg[k] /= max(n, 1)

    return avg


# ============================================================
# MAIN
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    results = evaluate_all_cities(
        model,
        args.msls_root,
        args.image_size,
        args.batch_size,
        device
    )

    avg = average_results(results)

    # ========================================================
    # REPORT
    # ========================================================
    print("\n====================")
    print("PER-CITY RESULTS")
    print("====================")

    for city, res in results.items():
        print(f"{city}: {res}")

    print("\n====================")
    print("GLOBAL AVERAGE")
    print("====================")
    print(avg)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--msls_root", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=512)

    args = parser.parse_args()

    main(args)