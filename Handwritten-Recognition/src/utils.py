"""
Utility functions for dataset mappings, metric reporting, and visualizations
"""

import os
import string
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def get_mnist_mapping() -> dict[int, str]:
    """Returns label mapping for MNIST (10 classes: 0-9)."""
    return {i: str(i) for i in range(10)}


def get_emnist_mapping() -> dict[int, str]:
    """
    Returns label mapping for EMNIST ByClass (62 classes):
    - 0-9: '0'-'9'
    - 10-35: 'A'-'Z'
    - 36-61: 'a'-'z'
    """
    mapping = {}
    for i in range(10):
        mapping[i] = str(i)
    for i, char in enumerate(string.ascii_uppercase):
        mapping[10 + i] = char
    for i, char in enumerate(string.ascii_lowercase):
        mapping[36 + i] = char
    return mapping


def plot_and_save_history(history, save_path: str = "training_history.png"):
    """Plots and saves accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history.get("accuracy", []), label="Train Accuracy", marker="o")
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
    axes[0].set_title("Model Accuracy Curve", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss
    axes[1].plot(history.history.get("loss", []), label="Train Loss", marker="o")
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="Val Loss", marker="o")
    axes[1].set_title("Model Loss Curve", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Training history saved to: {save_path}")


def plot_and_save_confusion_matrix(y_true, y_pred, target_names, save_path: str = "confusion_matrix.png", title="Confusion Matrix"):
    """Plots and saves confusion matrix heatmap."""
    num_classes = len(target_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    fig_size = (10, 8) if num_classes <= 10 else (16, 14)
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm,
        annot=(num_classes <= 10),
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(title, fontsize=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
    return cm

