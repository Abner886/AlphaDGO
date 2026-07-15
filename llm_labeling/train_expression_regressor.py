"""Train a transformer-based regressor on exported expression tensors.

"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



MASKED_LOGIT_SENTINEL = -1e7


@dataclass
class TrainingConfig:
    data_pt: Path
    scores_csv: Path
    save_path: Path
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    save_metadata: bool = True
    init_checkpoint: Optional[Path] = None


FILE_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = TrainingConfig(
    data_pt=FILE_DIR / "out" / "llm_exports" / "llm_export_100.pt",
    scores_csv=FILE_DIR / "out" / "llm_scores" / "output_100.csv",
    save_path=FILE_DIR / "out" / "checkpoints" / "expression_transformer_100.pt",
    device="cuda",
    seed=42,
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    d_model=128,
    n_heads=8,
    num_layers=4,
    ff_multiplier=4,
    dropout=0.1,
    grad_clip=1.0,
    val_ratio=0.1,
    test_ratio=0.1,
    save_metadata=True,
    init_checkpoint=None,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_suffix(path: Path) -> Optional[str]:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def _default_scores_path(data_pt: Path, suffix: Optional[str]) -> Path:
    base = data_pt.parent.parent if data_pt.parent.name == "llm_exports" else data_pt.parent
    name = suffix or "scores"
    return base / "llm_scores" / f"output_{name}.csv"


def _default_checkpoint_path(data_pt: Path, suffix: Optional[str]) -> Path:
    base = data_pt.parent.parent if data_pt.parent.name == "llm_exports" else data_pt.parent
    name = suffix or "model"
    return base / "checkpoints" / f"expression_transformer_{name}.pt"


class ExpressionDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
        super().__init__()
        self.features = features
        self.targets = targets
        self.weights = weights

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index], self.weights[index]


class ExpressionTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
    ):  # noqa: D401
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_multiplier * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        nn.init.normal_(self.positional, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        h = self.input_proj(x)
        h = h + self.positional[:, : h.size(1)]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        logits = self.out(pooled).squeeze(-1)
        return torch.sigmoid(logits)


def _load_scores(csv_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    import csv  # local import to avoid cost when unused

    scores: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "expression" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise ValueError("Score CSV must contain 'expression' and 'score' columns.")
        for row in reader:
            expr = (row.get("expression") or "").strip()
            if not expr:
                continue
            try:
                score_val = float(row.get("score", "nan"))
            except ValueError:
                continue
            score_val = float(np.clip(score_val, 0.0, 1.0))
            raw_weight = row.get("weight", "")
            try:
                weight_val = float(raw_weight)
            except (TypeError, ValueError):
                weight_val = 1.0
            weight_val = float(np.clip(weight_val, 0.0, np.finfo(np.float32).max))
            if weight_val == 0.0:
                weight_val = 1e-6
            scores[expr] = score_val
            weights[expr] = weight_val
    if not scores:
        raise ValueError("No valid scores found in CSV; cannot train model.")
    if not weights:
        raise ValueError("No valid weights found in CSV; cannot train model.")
    return scores, weights


def _load_tensor_data(data_pt: Path) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    payload = torch.load(data_pt, map_location="cpu")
    expressions = payload.get("expressions")
    masked_logits = payload.get("masked_logits")
    onehot_tensor = payload.get("onehot_tensor")
    if expressions is None or masked_logits is None or onehot_tensor is None:
        raise KeyError("PT file must contain 'expressions', 'masked_logits', 'onehot_tensor'.")
    masked_logits = torch.as_tensor(masked_logits, dtype=torch.float32)
    onehot_tensor = torch.as_tensor(onehot_tensor, dtype=torch.float32)
    if masked_logits.shape != onehot_tensor.shape:
        raise ValueError("masked_logits and onehot_tensor must have identical shapes.")
    return list(expressions), masked_logits, onehot_tensor


def _prepare_tensors(
    expressions: Sequence[str],
    masked_logits: torch.Tensor,
    onehot_tensor: torch.Tensor,
    scores: Dict[str, float],
    sample_weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    feature_list: List[torch.Tensor] = []
    target_list: List[float] = []
    weight_list: List[float] = []

    missing = 0
    for idx, expr in enumerate(expressions):
        score = scores.get(expr)
        if score is None:
            missing += 1
            continue
        weight = sample_weights.get(expr, 1.0)
        if weight <= 0.0:
            continue
        masked_block = masked_logits[idx]
        sanitized_block = torch.where(masked_block <= MASKED_LOGIT_SENTINEL, torch.zeros_like(masked_block), masked_block)
        feature = torch.cat([sanitized_block, onehot_tensor[idx]], dim=-1)
        feature_list.append(feature)
        target_list.append(float(score))
        weight_list.append(float(weight))

    if not feature_list:
        raise ValueError("No expressions had corresponding scores; aborting.")

    features = torch.stack(feature_list)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    targets = torch.tensor(target_list, dtype=torch.float32)
    weights_tensor = torch.tensor(weight_list, dtype=torch.float32)
    seq_len = features.size(1)
    input_dim = features.size(2)

    if missing > 0:
        print(f"Warning: skipped {missing} expressions without scores.")

    return features, targets, weights_tensor, seq_len, input_dim


def _split_indices(num_samples: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = int(math.floor(num_samples * val_ratio))
    test_size = int(math.floor(num_samples * test_ratio))
    train_size = num_samples - val_size - test_size
    if train_size <= 0:
        raise ValueError("Not enough samples for the chosen train/val/test split.")

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]
    return train_idx, val_idx, test_idx


def _build_dataloaders(
    features: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    cfg: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_idx, val_idx, test_idx = _split_indices(features.size(0), cfg.val_ratio, cfg.test_ratio, cfg.seed)

    train_ds = ExpressionDataset(features[train_idx], targets[train_idx], sample_weights[train_idx])
    val_ds = ExpressionDataset(features[val_idx], targets[val_idx], sample_weights[val_idx]) if val_idx.size > 0 else None
    test_ds = ExpressionDataset(features[test_idx], targets[test_idx], sample_weights[test_idx]) if test_idx.size > 0 else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False) if test_ds else None
    return train_loader, val_loader, test_loader


def _evaluate(model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> Tuple[float, float]:
    if loader is None:
        return float("nan"), float("nan")
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for features, targets, weights in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            preds = model(features)
            diff = preds - targets
            weighted_sq = (weights * diff.pow(2)).sum().item()
            weighted_abs = (weights * diff.abs()).sum().item()
            weight_sum = weights.sum().item()
            total_loss += weighted_sq
            total_mae += weighted_abs
            total_weight += weight_sum
    if total_weight <= 0.0:
        return float("nan"), float("nan")
    return total_loss / total_weight, total_mae / total_weight


def train(cfg: TrainingConfig) -> Dict[str, float]:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    expressions, masked_logits, onehot_tensor = _load_tensor_data(cfg.data_pt)
    scores, sample_weights = _load_scores(cfg.scores_csv)
    features, targets, weights_tensor, seq_len, input_dim = _prepare_tensors(expressions, masked_logits, onehot_tensor, scores, sample_weights)

    train_loader, val_loader, test_loader = _build_dataloaders(features, targets, weights_tensor, cfg)

    model = ExpressionTransformer(
        seq_len=seq_len,
        input_dim=input_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        ff_multiplier=cfg.ff_multiplier,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.init_checkpoint:
        init_path = Path(cfg.init_checkpoint)
        if init_path.is_file():
            init_payload = torch.load(init_path, map_location=device)
            init_state = init_payload.get("model_state") if isinstance(init_payload, dict) else None
            if init_state:
                missing, unexpected = model.load_state_dict(init_state, strict=False)
                if missing or unexpected:
                    print(f"Warning: init checkpoint missing {missing}, unexpected {unexpected}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for features_batch, targets_batch, weights_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            weights_batch = weights_batch.to(device)

            optimizer.zero_grad()
            preds = model(features_batch)
            squared_error = (preds - targets_batch).pow(2)
            weighted_loss = (weights_batch * squared_error).sum()
            denom = torch.clamp(weights_batch.sum(), min=1e-6)
            loss = weighted_loss / denom
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += weighted_loss.item()
            total_weight += weights_batch.sum().item()

        train_loss = total_loss / total_weight if total_weight else float("nan")
        val_loss, val_mae = _evaluate(model, val_loader, device)
        test_loss, test_mae = _evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mae={val_mae:.4f} test_loss={test_loss:.4f} test_mae={test_mae:.4f}"
        )

        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    os.makedirs(cfg.save_path.parent, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "config": {
            "seq_len": seq_len,
            "input_dim": input_dim,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "num_layers": cfg.num_layers,
            "ff_multiplier": cfg.ff_multiplier,
            "dropout": cfg.dropout,
            "score_stats": {
                "mean": float(targets.mean().item()),
                "std": float(targets.std(unbiased=False).item()),
            },
        },
    }
    if cfg.save_metadata:
        checkpoint["metadata"] = {
            "data_pt": str(cfg.data_pt),
            "scores_csv": str(cfg.scores_csv),
            "num_samples": int(features.size(0)),
        }
    torch.save(checkpoint, cfg.save_path)
    print(f"Saved checkpoint to {cfg.save_path}")

    final_val_loss, final_val_mae = _evaluate(model, val_loader, device)
    final_test_loss, final_test_mae = _evaluate(model, test_loader, device)
    return {
        "val_loss": final_val_loss,
        "val_mae": final_val_mae,
        "test_loss": final_test_loss,
        "test_mae": final_test_mae,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a transformer regressor on expression tensors.")
    parser.add_argument("--data-pt", help="Path to the .pt file from export_expr_dataset.py")
    parser.add_argument(
        "--scores-csv",
        help="CSV file with columns expression, score (0-1). If omitted, derives output_<n>.csv from data-pt filename llm_export_<n>.pt",
    )
    parser.add_argument(
        "--save-path",
        help="Where to store the trained model checkpoint. If omitted, derives expression_transformer_<n>.pt from data-pt filename llm_export_<n>.pt",
    )
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="Torch device to use (default: cuda)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.num_epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay, help="Weight decay coefficient")
    parser.add_argument("--d-model", type=int, default=DEFAULT_CONFIG.d_model, help="Transformer hidden size")
    parser.add_argument("--n-heads", type=int, default=DEFAULT_CONFIG.n_heads, help="Transformer attention heads")
    parser.add_argument("--layers", type=int, default=DEFAULT_CONFIG.num_layers, help="Number of Transformer encoder layers")
    parser.add_argument("--ff-multiplier", type=int, default=DEFAULT_CONFIG.ff_multiplier, help="Feed-forward size multiplier")
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG.dropout, help="Dropout probability")
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG.grad_clip, help="Gradient clipping threshold (0 disables)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_CONFIG.val_ratio, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_CONFIG.test_ratio, help="Test ratio")
    parser.add_argument("--no-metadata", action="store_true", help="Do not store metadata inside checkpoint")
    parser.add_argument("--init-checkpoint", help="Optional checkpoint providing initial weights for fine-tuning")

    args = parser.parse_args(argv)
    data_pt = Path(args.data_pt) if args.data_pt else DEFAULT_CONFIG.data_pt
    suffix = _infer_suffix(data_pt)
    scores_csv = Path(args.scores_csv) if args.scores_csv else _default_scores_path(data_pt, suffix)
    save_path = Path(args.save_path) if args.save_path else _default_checkpoint_path(data_pt, suffix)

    cfg = TrainingConfig(
        data_pt=data_pt,
        scores_csv=scores_csv,
        save_path=save_path,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        save_metadata=not args.no_metadata,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else DEFAULT_CONFIG.init_checkpoint,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    arg_list = list(argv) if argv is not None else sys.argv[1:]
    if arg_list:
        cfg = parse_args(arg_list)
    else:
        cfg = replace(DEFAULT_CONFIG)
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
