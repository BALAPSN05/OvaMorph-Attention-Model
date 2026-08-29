import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.dataset import build_dataloaders
from src.model import OvaMorphNet
from src.train_utils import run_epoch


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        workers=args.workers,
    )

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {classes}")

    model = OvaMorphNet(
        num_classes=len(classes),
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        num_prototypes=args.num_prototypes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1_macro']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "args": vars(args),
                "val_metrics": val_metrics,
            }, Path(args.output_dir) / "best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--num_prototypes", type=int, default=8)
    parser.add_argument("--no_pretrained", action="store_true")
    main(parser.parse_args())
