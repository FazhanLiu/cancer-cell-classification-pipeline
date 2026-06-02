import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from cancer_pipeline.metrics import classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int, amp: bool):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, (images, labels, _) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * grad_accum_steps * images.size(0)
    if len(loader) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    model.eval()
    ids, y_true, y_prob = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels, batch_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=1)[:, 1]
        total_loss += loss.item() * images.size(0)
        ids.extend(batch_ids)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    metrics = classification_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    frame = pd.DataFrame({"id": ids, "label": y_true, "prob_cancer": y_prob})
    return metrics, frame


def fit(model, loaders, criterion, optimizer, scheduler, device, config, output_dir: Path):
    train_cfg = config["train"]
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_auc = -1.0
    best_epoch = 0
    history = []
    patience = int(train_cfg.get("early_stopping_patience", 6))
    start = time.time()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
            amp=amp,
        )
        val_metrics, val_predictions = predict(model, loaders["val"], device, amp)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": val_metrics}, output_dir / "last.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": val_metrics}, output_dir / "best.pt")

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}")
        if epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    return {"best_auc": best_auc, "best_epoch": best_epoch, "seconds": time.time() - start}
