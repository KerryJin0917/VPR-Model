#!/usr/bin/env python3

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Sampler
import random
from collections import defaultdict

# ============================================================================
# Dataset
# ============================================================================

class VPRTrainDataset(Dataset):
    """Training dataset for VPR loaded from Parquet metadata.

    Groups images by location cluster (identity column) for contrastive learning.
    """

    def __init__(self, root: str, parquet_file: str = "train.parquet", image_size=(224, 224)):
        self.root = root
        df = pd.read_parquet(os.path.join(root, parquet_file))

        # Filter to train split if mixed
        if "split" in df.columns:
            train_df = df[df["split"] == "train"]
            if len(train_df) > 0:
                df = train_df

        # Use identity (cluster) as class label
        unique_ids = sorted(df["identity"].unique())
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)

        self.image_paths = df["image_path"].tolist()
        self.labels = [self.id_to_label[pid] for pid in df["identity"].values]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            # NEW: More aggressive augmentation for MSLS/GSV
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            # Handles different camera viewpoints in MSLS
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


class VPRTestDataset(Dataset):
    """Validation dataset for quick metric computation during training."""

    def __init__(self, root: str, parquet_file: str = "test.parquet", image_size=(224, 224), max_samples=200):
        self.root = root
        df = pd.read_parquet(os.path.join(root, parquet_file))

        # For datasets with role column (dataset_b)
        if "role" in df.columns:
            db_df = df[df["role"] == "database"].head(max_samples)
            q_df = df[df["role"] == "queries"].head(max_samples)
        else:
            db_df = df[df["split"] == "database"].head(max_samples)
            q_df = df[df["split"] == "queries"].head(max_samples)

        self.database_paths = [os.path.join(root, p) for p in db_df["image_path"].values]
        self.query_paths = [os.path.join(root, p) for p in q_df["image_path"].values]
        self.all_paths = self.database_paths + self.query_paths

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        image = Image.open(self.all_paths[idx]).convert("RGB")
        return self.transform(image), idx


# ============================================================================
# Loss Functions
# ============================================================================

class RKDLoss(nn.Module):
    """Relational Knowledge Distillation: Distance-wise distillation."""
    def __init__(self, w_dist=1.0):
        super().__init__()
        self.w_dist = w_dist

    def forward(self, student_feats, teacher_feats):
        # Calculate pair-wise Euclidean distances
        s_dist = torch.pdist(student_feats)
        t_dist = torch.pdist(teacher_feats)

        # Normalize distances so the scale of the embeddings doesn't matter
        s_dist = s_dist / (s_dist.mean() + 1e-7)
        t_dist = t_dist / (t_dist.mean() + 1e-7)

        # Loss is the structural difference between the two distance maps
        loss = F.smooth_l1_loss(s_dist, t_dist)
        return loss * self.w_dist

class ContrastiveLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(embeddings) / self.temperature
        return F.cross_entropy(logits, labels)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        dist_mat = 1 - torch.matmul(embeddings, embeddings.t())  # cosine distance

        labels = labels.unsqueeze(1)
        same = labels == labels.t()

        loss = 0.0
        valid = 0

        for i in range(embeddings.size(0)):
            pos_mask = same[i].clone()
            neg_mask = ~same[i]

            pos_mask[i] = False

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            hardest_pos = dist_mat[i][pos_mask].max()
            hardest_neg = dist_mat[i][neg_mask].min()

            loss += F.relu(hardest_pos - hardest_neg + self.margin)
            valid += 1

        return loss / max(valid, 1)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z, labels):
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.t

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float().to(z.device)

        # remove self similarity
        self_mask = torch.eye(sim.size(0), device=z.device)
        sim = sim.masked_fill(self_mask.bool(), -1e9)

        log_prob = F.log_softmax(sim, dim=1)

        pos_log_prob = (log_prob * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)

        return -pos_log_prob.mean()

# ============================================================================
# Model
# ============================================================================

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        # x is (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class TrainableModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Load DINOv2 without the head
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # New: GeM Pooling layer
        self.pooling = GeM()

        # DINOv2-Small (vits14) outputs 384 channels per spatial location
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 512),
            nn.LayerNorm(512), # Added for stability at large scale
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        self.teacher_proj = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        # Change: Use the output of the last block (n=1, index -1)
        # return_class_token=False ensures we get patch tokens for GeM pooling
        features_dict = self.backbone.forward_features(x)
        features = features_dict['x_norm_patchtokens'] # (B, 256, 384)

        # Reshape tokens back to spatial grid
        b, n, c = features.shape
        h = int(math.sqrt(n))
        w = n // h
        features = features.transpose(1, 2).reshape(b, c, h, w) # (B, 384, 16, 16)

        pooled = self.pooling(features)
        return self.head(pooled)

    def encode(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


# ============================================================================
# Training Loop
# ============================================================================
class PKBatchSampler(Sampler):
    def __init__(self, labels, P=8, K=4):
        self.labels = np.array(labels)
        self.P = P
        self.K = K

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.labels_unique = list(self.label_to_indices.keys())

    def __iter__(self):
        indices = []
        # Ensure we don't request more classes than exist
        num_classes_to_sample = min(self.P, len(self.labels_unique))

        # We process in full epochs
        for _ in range(len(self.labels) // (self.P * self.K)):
            # Fallback to current classes if we don't have P
            chosen_p = random.sample(self.labels_unique, num_classes_to_sample)

            # If we don't have enough classes, duplicate some to fill the P requirement
            while len(chosen_p) < self.P:
                chosen_p.append(random.choice(chosen_p))

            batch = []
            for p in chosen_p:
                l = self.label_to_indices[p]
                # If a specific place doesn't have K images, sample with replacement
                if len(l) >= self.K:
                    batch.extend(random.sample(l, self.K))
                else:
                    batch.extend(random.choices(l, k=self.K))

            indices.append(batch)
        return iter(indices)

    def __len__(self):
        return len(self.labels) // (self.P * self.K)

class MemoryBank:
    def __init__(self, size, dim, device):
        self.size = size
        self.bank = torch.randn(size, dim).to(device)
        self.bank = F.normalize(self.bank, dim=1)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feats):
        b = feats.size(0)
        end = self.ptr + b

        if end <= self.size:
            self.bank[self.ptr:end] = feats.detach()
        else:
            first = self.size - self.ptr
            self.bank[self.ptr:] = feats[:first].detach()
            self.bank[:end % self.size] = feats[first:].detach()

        self.ptr = end % self.size

    def get(self):
        return self.bank

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    parquet_file = "train.parquet" if os.path.exists(os.path.join(args.data_root, "train.parquet")) else "test.parquet"
    dataset = VPRTrainDataset(args.data_root, parquet_file=parquet_file,
                              image_size=(args.image_size, args.image_size))
    sampler = PKBatchSampler(dataset.labels, P=16, K=4)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Training set: {len(dataset)} images, {dataset.num_classes} places")

    memory_bank = MemoryBank(size=8192, dim=args.embedding_dim, device=device)

    # Model (Student)
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)

    # Load Teacher (e.g., DINOv2-Large)
    print("Loading Teacher model (DINOv2-Large)...")
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    teacher.eval() # CRITICAL: Teacher is frozen
    for param in teacher.parameters():
        param.requires_grad = False


    # Loss
    if args.loss == "triplet":
        print("Using Triplet Loss")
        criterion = BatchHardTripletLoss(margin=args.margin).to(device)
    elif args.loss == "contrastive":
        print("Using Contrastive Loss")
        criterion = ContrastiveLoss(embedding_dim=args.embedding_dim, num_classes=dataset.num_classes).to(device)
    else: # Default to InfoNCE
        print("Using InfoNCE Loss")
        criterion = InfoNCELoss(temperature=0.07).to(device)

        # RKD Loss (Add this separately)
    rkd_criterion = None
    if args.use_distill:
        rkd_criterion = RKDLoss(w_dist=1.0).to(device)

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    # Only grab loss params if the loss actually has them (like ContrastiveLoss)
    loss_params = []
    if isinstance(criterion, ContrastiveLoss):
        loss_params = list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.01},
        {"params": head_params, "lr": args.lr},
        {"params": model.teacher_proj.parameters(), "lr": args.lr},
        {"params": loss_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # LR Scheduler: cosine annealing with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = len(dataloader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(args.epochs):
        model.train()
        if hasattr(criterion, 'train'):
            criterion.train()

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # =========================
            # Teacher forward (frozen)
            # =========================
            with torch.no_grad(), torch.amp.autocast("cuda"):
                t_feats = teacher.forward_features(images)["x_norm_clstoken"]

            teacher_embeddings = model.teacher_proj(t_feats.float())
            teacher_embeddings = F.normalize(teacher_embeddings, dim=1)

            # =========================
            # Student forward
            # =========================
            with torch.amp.autocast("cuda"):
                embeddings = model(images)
                embeddings = F.normalize(embeddings, dim=1)

                mem = memory_bank.get()
                all_feats = torch.cat([embeddings, mem], dim=0)
                logits = torch.matmul(embeddings, all_feats.T) / 0.07

                B = embeddings.size(0)
                self_mask = torch.eye(B, device=device).bool()
                logits[:, :B] = logits[:, :B].masked_fill(self_mask, torch.finfo(logits.dtype).min)

                labels_exp = labels.unsqueeze(1)
                pos_mask = (labels_exp == labels_exp.T).float().to(device)

                pos_mask = torch.cat(
                    [pos_mask, torch.zeros((B, mem.size(0)), device=device)],
                    dim=1
                )

                log_prob = F.log_softmax(logits, dim=1)
                base_loss = -(log_prob * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
                base_loss = base_loss.mean()

                if args.loss == "infonce":
                    loss = base_loss
                else:
                    loss = criterion(embeddings, labels)

                if args.use_distill:
                    rkd_loss = rkd_criterion(embeddings, teacher_embeddings)
                    loss = loss + 10.0 * rkd_loss

            # =========================
            # Backprop
            # =========================
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Memory update
            with torch.no_grad():
                memory_bank.update(embeddings.detach())

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

    # =========================
    # NOW OUTSIDE LOOP (IMPORTANT)
    # =========================
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'embedding_dim': args.embedding_dim,
        }, save_dir / "best_model.pth")
        print(f"Saved best model (loss={avg_loss:.4f})")

    if (epoch + 1) % args.save_every == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            'embedding_dim': args.embedding_dim,
        }, save_dir / f"checkpoint_epoch{epoch+1}.pth")


# ============================================================================
# Prediction Generation
# ============================================================================

class ImageDataset(Dataset):
    """Simple image dataset for inference."""

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
        path = self.image_paths[idx] # Use the path as-is
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx

def denormalize(x):
    mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
    return x * std + mean

def encode_images_multiscale(model, dataset, device, scales=[0.707, 1.0, 1.414]):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def normalize(x):
        return (x - mean) / std

    all_feats = []

    with torch.inference_mode():
        for images, idx in tqdm(loader, desc="Multi-scale encoding"):
            images = images.to(device)

            feats = []

            for s in scales:
                if s != 1.0:
                    size = int(224 * s)
                    x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
                else:
                    x = images


                feats.append(model.encode(x))

            feats = torch.stack(feats).mean(dim=0)
            all_feats.append(feats.cpu().numpy())

    return np.vstack(all_feats)


# ============================================================================
# Simplified Loader (No GitHub Dependencies)
# ============================================================================

def load_db_queries(root, dataset_name):
    import os
    import pandas as pd

    csv_path = os.path.join(root, "Dataframes", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)

    image_dir = os.path.join(root, "Images", dataset_name)

    # panoid → filepath
    panoid_to_path = {}
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg"):
            panoid = fname.split("_")[-1].replace(".jpg", "")
            panoid_to_path[panoid] = os.path.join(image_dir, fname)

    df["image_path"] = df["panoid"].map(panoid_to_path)
    df = df.dropna(subset=["image_path"])

    # split by place_id
    db_list = []
    query_list = []

    for place_id, group in df.groupby("place_id"):
        if len(group) < 2:
            continue

        group = group.sample(frac=1, random_state=42)
        query_list.append(group.iloc[0])
        db_list.extend(group.iloc[1:].to_dict("records"))

    db_df = pd.DataFrame(db_list)
    query_df = pd.DataFrame(query_list)

    db_paths = db_df["image_path"].tolist()
    query_paths = query_df["image_path"].tolist()

    db_labels = db_df["place_id"].values
    query_labels = query_df["place_id"].values

    print(f"--- Evaluation Mode ---")
    print(f"Dataset: {dataset_name}")
    print(f"Database images: {len(db_paths)}")
    print(f"Query images: {len(query_paths)}")

    return db_paths, query_paths, db_labels, query_labels

def compute_recall_at_k(rankings, db_labels, query_labels, ks=[1,5,10,20]):
    recalls = {k: 0 for k in ks}

    for i, q_label in enumerate(query_labels):
        retrieved = rankings[i]

        for k in ks:
            top_k = retrieved[:k]
            if any(db_labels[j] == q_label for j in top_k):
                recalls[k] += 1

    for k in ks:
        recalls[k] /= len(query_labels)

    return recalls

def rerank_topk(query_emb, db_emb, rankings, top_m=50):
    reranked = []

    for i in range(len(query_emb)):
        topk_idx = rankings[i][:top_m]

        q = query_emb[i]
        candidates = db_emb[topk_idx]

        sims = np.dot(candidates, q)
        new_order = np.argsort(-sims)

        reranked.append(topk_idx[new_order])

    return np.array(reranked)

def predict(args):
    """Generate prediction CSV from a trained checkpoint.

    Loads the trained model, encodes database and query images, computes
    cosine similarity, and outputs ranked database indices per query.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load database/query paths
    db_paths, query_paths, db_labels, query_labels = load_db_queries(args.dataset_root, args.dataset_name)
    print(f"Database: {len(db_paths)}, Queries: {len(query_paths)}")

    # Encode
    img_size = (args.image_size, args.image_size)
    db_dataset = ImageDataset(args.dataset_root, db_paths, img_size)
    query_dataset = ImageDataset(args.dataset_root, query_paths, img_size)
    db_emb = encode_images_multiscale(model, db_dataset, device)
    query_emb = encode_images_multiscale(model, query_dataset, device)

    # Compute rankings
    print("Computing rankings...")
    similarity = np.matmul(query_emb, db_emb.T)
    full_rankings = np.argsort(-similarity, axis=1)

    rankings = rerank_topk(query_emb, db_emb, full_rankings, top_m=50)
    rankings = rankings[:, :args.top_k]
    # Compute Recall@K
    recalls = compute_recall_at_k(rankings, db_labels, query_labels)

    print("\n--- Recall@K ---")
    for k, v in recalls.items():
        print(f"Recall@{k}: {v:.4f}")

    # Save predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_idx in range(len(query_paths)):
        ranked_str = ",".join(str(x) for x in rankings[q_idx])
        rows.append({"query_index": q_idx, "ranked_database_indices": ranked_str})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output} ({len(rows)} queries)")


def main():
    parser = argparse.ArgumentParser(description="VPR Training Example")

    # Mode
    parser.add_argument("--predict", action="store_true", help="Generate predictions from a checkpoint")

    # Training args
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_b", help="Dataset root for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "triplet", "infonce"], help="Loss function")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--use_distill", action="store_true", help="Use RKD distillation")

    # Prediction args
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Checkpoint path for prediction")
    parser.add_argument("--dataset_root", type=str, help="Dataset root for prediction")
    parser.add_argument("--dataset_name", type=str, help="Dataset name for prediction")
    parser.add_argument("--output", type=str, default="predictions/dataset_a.csv", help="Output CSV path for predictions")
    parser.add_argument("--top_k", type=int, default=20, help="Number of ranked results per query")

    # Shared args
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if args.predict:
        if not args.dataset_root:
            parser.error("--dataset_root is required for prediction mode")
        if not args.dataset_name:
            parser.error("--dataset_name is required for prediction mode")
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
