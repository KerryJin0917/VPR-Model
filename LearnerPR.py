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
    """NT-Xent (InfoNCE) contrastive loss for place recognition."""

    def __init__(self, embedding_dim, num_classes, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings, labels):
        # Cross-entropy with temperature-scaled logits
        logits = self.classifier(F.normalize(embeddings, p=2, dim=1)) / self.temperature
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """Online hard triplet mining loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = 1 - torch.mm(embeddings, embeddings.t())

        labels = labels.unsqueeze(0)
        same_identity = labels == labels.t()

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(embeddings.size(0)):
            pos_mask = same_identity[i].clone()
            pos_mask[i] = False
            neg_mask = ~same_identity[i]

            if pos_mask.any() and neg_mask.any():
                hardest_pos = dist_mat[i][pos_mask].max()
                hardest_neg = dist_mat[i][neg_mask].min()
                loss += F.relu(hardest_pos - hardest_neg + self.margin)
                count += 1

        return loss / max(count, 1)


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

    def forward(self, x):
        # Change: Get spatial features instead of just CLS token
        # For DINOv2, we get features from the last layer
        features = self.backbone.get_intermediate_layers(x, n=1)[0] # (B, 256, 384)

        # Reshape tokens back to spatial grid (224/14 = 16)
        b, n, c = features.shape
        h = w = int(math.sqrt(n))
        features = features.transpose(1, 2).reshape(b, c, h, w) # (B, 384, 16, 16)

        pooled = self.pooling(features)
        return self.head(pooled)

    def encode(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    parquet_file = "train.parquet" if os.path.exists(os.path.join(args.data_root, "train.parquet")) else "test.parquet"
    dataset = VPRTrainDataset(args.data_root, parquet_file=parquet_file,
                              image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    print(f"Training set: {len(dataset)} images, {dataset.num_classes} places")

    # Model (Student)
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)

    # Load Teacher (e.g., DINOv2-Large)
    print("Loading Teacher model (DINOv2-Large)...")
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    teacher.eval() # CRITICAL: Teacher is frozen
    for param in teacher.parameters():
        param.requires_grad = False


    # Loss
    if args.loss == "contrastive":
        criterion = ContrastiveLoss(args.embedding_dim, dataset.num_classes).to(device)
    elif args.loss == "triplet":
        criterion = TripletLoss(margin=args.margin)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

        # RKD Loss (Add this separately)
    rkd_criterion = None
    if args.use_distill:
        rkd_criterion = RKDLoss(w_dist=1.0).to(device)

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    # Only grab loss params if the loss actually has them (like ContrastiveLoss)
    loss_params = list(criterion.parameters()) if hasattr(criterion, 'parameters') else []

    optimizer = torch.optim.AdamW([
       {"params": backbone_params, "lr": args.lr * 0.01}, # Low LR for DINOv2
      {"params": head_params, "lr": args.lr},           # Standard LR for Head
      {"params": loss_params, "lr": args.lr},           # Standard LR for Loss (if exists)
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

    for epoch in range(args.epochs):
        model.train()
        if hasattr(criterion, 'train'):
            criterion.train()

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # 1. New: Get Teacher embeddings (Frozen, no gradients needed)
            with torch.no_grad():
                teacher_embeddings = teacher(images)
                teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)

            #2. Existing: Get Student embeddings
            embeddings = model(images)

            # 3. New: Generate normalized version for RKD loss computation
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)

            # 4. Existing: Standard cross-entropy / contrastive loss
            base_loss = criterion(embeddings, labels)

            # 5. New: Distillation Loss calculation
            # Only calculate if a distillation flag is active (if you set one up)
            if args.use_distill:
                rkd_loss = rkd_criterion(embeddings_norm, teacher_embeddings)
              # alpha balances the two losses; 10.0 is a typical starting value
                alpha = 10.0
                loss = base_loss + (alpha * rkd_loss)
            else:
                loss = base_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

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
            print(f"  Saved best model (loss={avg_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
            }, save_dir / f"checkpoint_epoch{epoch+1}.pth")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


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
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


def encode_images(model, dataset, batch_size, num_workers, device):
    """Encode all images and return L2-normalized embeddings in order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    emb_list, idx_list = [], []
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Encoding"):
            emb = model.encode(images.to(device))
            emb_list.append(emb.cpu().numpy())
            idx_list.append(indices.numpy())
    embeddings = np.vstack(emb_list)
    indices = np.concatenate(idx_list)
    return embeddings[np.argsort(indices)]


def load_db_queries(root, dataset_name):
    csv_path = os.path.join(root, "Dataframes", f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)

    # Handle the database/query split
    if "role" in df.columns:
        db_df, q_df = df[df["role"] == "database"], df[df["role"] == "queries"]
    else:
        mid = len(df) // 2
        db_df, q_df = df.iloc[:mid], df.iloc[mid:]

    def get_paths(dataframe):
        paths = []
        for _, row in dataframe.iterrows():
            # Construct the filename using the metadata in the CSV
            filename = (
                f"{dataset_name}_{int(row['place_id']):07d}_{row['year']}_"
                f"{int(row['month']):02d}_{int(row['northdeg'])}_"
                f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
            )
            # Add the relative path to the city folder
            paths.append(os.path.join("Images", dataset_name, filename))
        return paths

    db_paths = get_paths(db_df)
    query_paths = get_paths(q_df)

    return db_paths, query_paths


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
    db_paths, query_paths = load_db_queries(args.dataset_root, args.dataset_name)
    print(f"Database: {len(db_paths)}, Queries: {len(query_paths)}")

    # Encode
    img_size = (args.image_size, args.image_size)
    db_emb = encode_images(model, ImageDataset(args.dataset_root, db_paths, img_size),
                           args.batch_size, args.num_workers, device)
    query_emb = encode_images(model, ImageDataset(args.dataset_root, query_paths, img_size),
                              args.batch_size, args.num_workers, device)

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


def main():
    parser = argparse.ArgumentParser(description="VPR Training Example")

    # Mode
    parser.add_argument("--predict", action="store_true", help="Generate predictions from a checkpoint")

    # Training args
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_b", help="Dataset root for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "triplet"], help="Loss function")
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
