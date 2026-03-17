import os
import json
import numpy as np
from tqdm import tqdm

import torch

from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import seaborn as sns


TASK_PROXEMICS = "proxemics"
TASK_RELATIONSHIP = "relationship"
TASK_MULTITASK = "multitask"

PROXEMICS_CLASS_NAMES = [
    "HAND_HAND",
    "HAND_SHOULDER",
    "SHOULDER_SHOULDER",
    "HAND_TORSO",
    "HAND_ELBOW",
    "ELBOW_SHOULDER",
]

RELATIONSHIP_CLASS_NAMES = [
    "Friends",
    "Family",
    "Couple",
    "Professional",
    "Commercial",
    "No Relation",
]


def load_saved_config(model_dir):
    config_path_json = os.path.join(model_dir, "config.json")
    config_path_yaml = os.path.join(model_dir, "config.yaml")

    if os.path.exists(config_path_json):
        with open(config_path_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.exists(config_path_yaml):
        import yaml
        with open(config_path_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"No config.json or config.yaml found inside {model_dir}")


def load_best_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def evaluate_on_test(task, model, test_loader, device):
    model.eval()

    prox_preds, prox_gts = [], []
    rel_preds, rel_gts = [], []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")

        for batch in test_pbar:
            if task == TASK_PROXEMICS:
                X, y = batch
                X = [x.to(device) for x in X]

                logits = model(X)
                y_prob = torch.sigmoid(logits).cpu()

                prox_preds.append(y_prob)
                prox_gts.append(y.cpu())

            elif task == TASK_RELATIONSHIP:
                X, (_, y) = batch
                X = [x.to(device) for x in X]
                y = y.to(device).long()

                logits = model(X)
                pred = torch.argmax(logits, dim=1).cpu()

                rel_preds.append(pred)
                rel_gts.append(y.cpu())

            elif task == TASK_MULTITASK:
                X, (y_prox, y_rel) = batch
                X = [x.to(device) for x in X]
                y_prox = y_prox.float().to(device)
                y_rel = y_rel.long().to(device)

                logits = model(X)

                probs_prox = torch.sigmoid(logits["proxemics"]).cpu()
                pred_rel = torch.argmax(logits["relationship"], dim=1).cpu()

                prox_preds.append(probs_prox)
                prox_gts.append(y_prox.cpu())
                rel_preds.append(pred_rel)
                rel_gts.append(y_rel.cpu())

            else:
                raise ValueError(f"Unknown task: {task}")

    results = {}

    if task in [TASK_PROXEMICS, TASK_MULTITASK]:
        y_pred_prox = torch.cat(prox_preds).numpy()
        y_true_prox = torch.cat(prox_gts).numpy()

        aps_test = [
            average_precision_score(y_true_prox[:, i], y_pred_prox[:, i])
            for i in range(6)
        ]
        map_test = float(np.nanmean(aps_test))

        results["proxemics"] = {
            "AP_per_class": dict(zip(PROXEMICS_CLASS_NAMES, aps_test)),
            "mAP": map_test,
        }

    if task in [TASK_RELATIONSHIP, TASK_MULTITASK]:
        y_pred_rel = torch.cat(rel_preds).numpy()
        y_true_rel = torch.cat(rel_gts).numpy()

        acc_test = accuracy_score(y_true_rel, y_pred_rel)
        f1m_test = f1_score(y_true_rel, y_pred_rel, average="macro", zero_division=0)

        num_classes = len(RELATIONSHIP_CLASS_NAMES)
        cm = confusion_matrix(y_true_rel, y_pred_rel, labels=np.arange(num_classes))

        prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
            y_true_rel,
            y_pred_rel,
            labels=np.arange(num_classes),
            average=None,
            zero_division=0,
        )

        per_class = {}
        for i, name in enumerate(RELATIONSHIP_CLASS_NAMES):
            per_class[name] = {
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(supp_c[i]),
            }

        results["relationship"] = {
            "accuracy": acc_test,
            "f1_macro": f1m_test,
            "confusion_matrix": cm,
            "per_class": per_class,
        }

    return results


def print_test_results(task, results):
    if task in [TASK_PROXEMICS, TASK_MULTITASK]:
        print("\n── Test AP per class ──")
        for name, ap in results["proxemics"]["AP_per_class"].items():
            print(f"{name:20s}: {ap:.4f}")
        print(f"mAP_proxemics: {results['proxemics']['mAP']:.4f}")

    if task in [TASK_RELATIONSHIP, TASK_MULTITASK]:
        acc = results["relationship"]["accuracy"]
        f1m = results["relationship"]["f1_macro"]

        print(f"\nTest Accuracy: {acc:.4f} | Test F1_macro: {f1m:.4f}\n")
        print("Per-class metrics:")
        print(f"{'Class':>12s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'N':>6s}")

        for name, metrics in results["relationship"]["per_class"].items():
            print(
                f"{name[:12]:>12s}  "
                f"{metrics['precision']:7.4f}  "
                f"{metrics['recall']:7.4f}  "
                f"{metrics['f1']:7.4f}  "
                f"{metrics['support']:6d}"
            )


def save_confusion_matrix_figure(confusion_matrix_array, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=RELATIONSHIP_CLASS_NAMES,
        yticklabels=RELATIONSHIP_CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_test_results_json(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {}

        for subkey, subvalue in value.items():
            if subkey == "confusion_matrix":
                serializable_results[key][subkey] = subvalue.tolist()
            else:
                serializable_results[key][subkey] = subvalue

    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)


def log_test_results_to_wandb_same_run(task, results, wandb_run, output_dir=None):
    if wandb_run is None:
        return

    if task in [TASK_PROXEMICS, TASK_MULTITASK]:
        wandb_run.log({
            "test_AP_proxemics": results["proxemics"]["AP_per_class"],
            "test_mAP_proxemics": results["proxemics"]["mAP"],
        })

    if task in [TASK_RELATIONSHIP, TASK_MULTITASK]:
        wandb_run.log({
            "test_accuracy_relationship": results["relationship"]["accuracy"],
            "test_f1_macro_relationship": results["relationship"]["f1_macro"],
        })

        table_data = []
        for name, m in results["relationship"]["per_class"].items():
            table_data.append([
                name,
                m["precision"],
                m["recall"],
                m["f1"],
                m["support"],
            ])


        cm = results["relationship"]["confusion_matrix"]