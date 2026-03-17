import os
import math
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import average_precision_score, accuracy_score, f1_score


TASK_PROXEMICS = "proxemics"
TASK_RELATIONSHIP = "relationship"
TASK_MULTITASK = "multitask"


def build_criteria(task, train_set=None, train_sampler=None, device="cuda"):
    if task == TASK_PROXEMICS:
        return {
            "proxemics": nn.BCEWithLogitsLoss()
        }

    elif task == TASK_RELATIONSHIP:
        return {
            "relationship": nn.CrossEntropyLoss()
        }

    elif task == TASK_MULTITASK:
        criterion_prox = nn.BCEWithLogitsLoss()

        y_rel_all = np.array([int(s["relationship"]) for s in train_set], dtype=np.int64)
        train_sampler.set_epoch(1)
        epoch1_idx = np.fromiter(iter(train_sampler), dtype=np.int64)

        rel_counts_eff = np.bincount(y_rel_all[epoch1_idx], minlength=6).astype(np.float64)
        assert (rel_counts_eff > 0).all(), "Some relationship classes have 0 samples in epoch 1."

        weights = rel_counts_eff.sum() / rel_counts_eff
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights.astype(np.float32), device=device)

        criterion_rel = nn.CrossEntropyLoss(weight=class_weight)

        return {
            "proxemics": criterion_prox,
            "relationship": criterion_rel
        }

    else:
        raise ValueError(f"Unknown task: {task}")


def build_optimizer_and_scheduler(model, learning_rate, weight_decay, task):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if task == TASK_MULTITASK:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=4)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=4)

    return optimizer, scheduler


def train_one_epoch(
    task,
    model,
    train_loader,
    train_sampler,
    optimizer,
    criteria,
    device,
    batch_size,
    epoch,
    num_epochs,
    alpha=1.0
):
    model.train()
    running_loss = 0.0

    train_sampler.set_epoch(epoch)
    _ = list(iter(train_sampler))
    num_batches = math.ceil(len(train_sampler.last_epoch_indices) / batch_size)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", total=num_batches)

    for batch in pbar:
        optimizer.zero_grad()

        if task == TASK_PROXEMICS:
            X, y = batch
            X = [x.to(device) for x in X]
            y = y.to(device).float()

            logits = model(X)
            loss = criteria["proxemics"](logits, y)

        elif task == TASK_RELATIONSHIP:
            X, (_, y) = batch
            X = [x.to(device) for x in X]
            y = y.to(device).long()

            logits = model(X)
            loss = criteria["relationship"](logits, y)

        elif task == TASK_MULTITASK:
            X, (y_prox, y_rel) = batch
            X = [x.to(device) for x in X]
            y_prox = y_prox.to(device).float()
            y_rel = y_rel.to(device).long()

            logits = model(X)
            loss_prox = criteria["proxemics"](logits["proxemics"], y_prox)
            loss_rel = criteria["relationship"](logits["relationship"], y_rel)
            loss = loss_prox + alpha * loss_rel

        else:
            raise ValueError(f"Unknown task: {task}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (pbar.n + 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    train_loss = running_loss / len(train_loader)
    return train_loss


def validate_one_epoch(task, model, val_loader, criteria, device, alpha=1.0):
    model.eval()
    val_loss = 0.0

    prox_preds, prox_gts = [], []
    rel_preds, rel_gts = [], []

    with torch.no_grad():
        for batch in val_loader:
            if task == TASK_PROXEMICS:
                X, y = batch
                X = [x.to(device) for x in X]
                y = y.to(device).float()

                logits = model(X)
                loss = criteria["proxemics"](logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu()
                prox_preds.append(probs)
                prox_gts.append(y.cpu())

            elif task == TASK_RELATIONSHIP:
                X, (_, y) = batch
                X = [x.to(device) for x in X]
                y = y.to(device).long()

                logits = model(X)
                loss = criteria["relationship"](logits, y)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1).cpu()
                rel_preds.append(pred)
                rel_gts.append(y.cpu())

            elif task == TASK_MULTITASK:
                X, (y_prox, y_rel) = batch
                X = [x.to(device) for x in X]
                y_prox = y_prox.to(device).float()
                y_rel = y_rel.to(device).long()

                logits = model(X)

                loss_prox = criteria["proxemics"](logits["proxemics"], y_prox)
                loss_rel = criteria["relationship"](logits["relationship"], y_rel)
                loss = loss_prox + alpha * loss_rel
                val_loss += loss.item()

                probs_prox = torch.sigmoid(logits["proxemics"]).cpu()
                prox_preds.append(probs_prox)
                prox_gts.append(y_prox.cpu())

                pred_rel = torch.argmax(logits["relationship"], dim=1).cpu()
                rel_preds.append(pred_rel)
                rel_gts.append(y_rel.cpu())

            else:
                raise ValueError(f"Unknown task: {task}")

    val_loss = val_loss / len(val_loader)
    metrics = {"val_loss": val_loss}

    if task in [TASK_PROXEMICS, TASK_MULTITASK]:
        y_pred_prox = torch.cat(prox_preds).numpy()
        y_true_prox = torch.cat(prox_gts).numpy()
        APs = [average_precision_score(y_true_prox[:, i], y_pred_prox[:, i]) for i in range(6)]
        metrics["val_mAP"] = float(np.nanmean(APs))

    if task in [TASK_RELATIONSHIP, TASK_MULTITASK]:
        y_pred_rel = torch.cat(rel_preds).numpy()
        y_true_rel = torch.cat(rel_gts).numpy()
        metrics["val_accuracy"] = accuracy_score(y_true_rel, y_pred_rel)
        metrics["val_f1_macro"] = f1_score(
            y_true_rel, y_pred_rel, average="macro", zero_division=0
        )

    return metrics


def get_monitor_config(task):
    if task == TASK_PROXEMICS:
        return {
            "checkpoint_metric": "val_mAP",
            "checkpoint_mode": "max",
            "early_stop_metric": "val_loss",
            "early_stop_mode": "min",
            "scheduler_metric": "val_loss",
        }

    elif task == TASK_RELATIONSHIP:
        return {
            "checkpoint_metric": "val_f1_macro",
            "checkpoint_mode": "max",
            "early_stop_metric": "val_loss",
            "early_stop_mode": "min",
            "scheduler_metric": "val_loss",
        }

    elif task == TASK_MULTITASK:
        return {
            "checkpoint_metric": "val_mAP",
            "checkpoint_mode": "max",
            "early_stop_metric": "val_mAP",
            "early_stop_mode": "max",
            "scheduler_metric": "val_mAP",
        }

    else:
        raise ValueError(f"Unknown task: {task}")


def is_better(current, best, mode):
    if best is None:
        return True
    if mode == "max":
        return current > best
    elif mode == "min":
        return current < best
    else:
        raise ValueError(f"Unknown mode: {mode}")


def save_checkpoint(model, task, epoch, metrics, checkpoint_path):
    payload = {
        "model": model.state_dict(),
        "task": task,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)


def save_training_config(config_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def format_metrics_for_print(task, train_loss, metrics):
    if task == TASK_PROXEMICS:
        return (
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_mAP={metrics['val_mAP']:.4f}"
        )

    elif task == TASK_RELATIONSHIP:
        return (
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"val_f1_macro={metrics['val_f1_macro']:.4f}"
        )

    elif task == TASK_MULTITASK:
        return (
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_mAP={metrics['val_mAP']:.4f} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"val_f1_macro={metrics['val_f1_macro']:.4f}"
        )

    else:
        raise ValueError(f"Unknown task: {task}")


def fit(
    task,
    model,
    train_loader,
    val_loader,
    train_sampler,
    train_set,
    device,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    weight_decay=0.01,
    alpha=1.0,
    early_stop_patience=8,
    wandb_run=None,
    config_to_save=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if config_to_save is not None:
        save_training_config(config_to_save, output_dir)

    criteria = build_criteria(
        task=task,
        train_set=train_set,
        train_sampler=train_sampler,
        device=device
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        task=task
    )

    monitor_cfg = get_monitor_config(task)

    ckpt_best = os.path.join(output_dir, "model_best.pt")
    ckpt_last = os.path.join(output_dir, "model_last.pt")

    best_checkpoint_value = None
    best_earlystop_value = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n==== Epoch {epoch}/{epochs} ====")

        train_loss = train_one_epoch(
            task=task,
            model=model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            optimizer=optimizer,
            criteria=criteria,
            device=device,
            batch_size=batch_size,
            epoch=epoch,
            num_epochs=epochs,
            alpha=alpha,
        )

        val_metrics = validate_one_epoch(
            task=task,
            model=model,
            val_loader=val_loader,
            criteria=criteria,
            device=device,
            alpha=alpha,
        )

        print(format_metrics_for_print(task, train_loss, val_metrics), flush=True)

        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_dict.update(val_metrics)

        if wandb_run is not None:
            wandb_run.log(log_dict)

        scheduler_metric_value = val_metrics[monitor_cfg["scheduler_metric"]]
        scheduler.step(scheduler_metric_value)

        save_checkpoint(model, task, epoch, val_metrics, ckpt_last)

        checkpoint_metric_name = monitor_cfg["checkpoint_metric"]
        checkpoint_metric_value = val_metrics[checkpoint_metric_name]
        checkpoint_mode = monitor_cfg["checkpoint_mode"]

        if is_better(checkpoint_metric_value, best_checkpoint_value, checkpoint_mode):
            best_checkpoint_value = checkpoint_metric_value
            save_checkpoint(model, task, epoch, val_metrics, ckpt_best)
            print(
                f"  ✔️ Best model saved "
                f"({checkpoint_metric_name}={checkpoint_metric_value:.4f})"
            )

        early_metric_name = monitor_cfg["early_stop_metric"]
        early_metric_value = val_metrics[early_metric_name]
        early_mode = monitor_cfg["early_stop_mode"]

        if is_better(early_metric_value, best_earlystop_value, early_mode):
            best_earlystop_value = early_metric_value
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break