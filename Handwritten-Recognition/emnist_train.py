"""
EMNIST ByClass Handwritten Digit & Character Recognition Training Script

This script loads the official EMNIST 'byclass' dataset (62 classes: 0-9, A-Z, a-z)
directly from local ubyte gzip files in data/emnist/, balances the samples across
all 62 categories, preprocesses and normalizes the images, builds and trains a
high-performance Convolutional Neural Network (CNN), evaluates classification metrics,
and saves the trained model to 'emnist_model.keras' and 'models/emnist_model.keras'.
"""

import os
import gzip
import struct
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

NUM_CLASSES = 62


def get_emnist_byclass_mapping():
    """
    Build and return label mapping dictionary and list for EMNIST ByClass (62 classes):
    - 0-9:   Digits '0' through '9' (classes 0-9)
    - 10-35: Uppercase letters 'A' through 'Z' (classes 10-35)
    - 36-61: Lowercase letters 'a' through 'z' (classes 36-61)
    """
    mapping = {}
    target_names = []

    # 0 - 9: Digits '0' - '9'
    for i in range(10):
        char = str(i)
        mapping[i] = char
        target_names.append(char)

    # 10 - 35: Uppercase 'A' - 'Z'
    for i, char in enumerate(string.ascii_uppercase):
        class_idx = 10 + i
        mapping[class_idx] = char
        target_names.append(char)

    # 36 - 61: Lowercase 'a' - 'z'
    for i, char in enumerate(string.ascii_lowercase):
        class_idx = 36 + i
        mapping[class_idx] = char
        target_names.append(char)

    return mapping, target_names


def load_ubyte_images_and_labels(images_path, labels_path, max_per_class=2000):
    """
    Reads IDX-format gzip binary images and labels, extracts balanced samples per class,
    and returns upright normalized arrays.
    """
    print(f"Reading dataset: {os.path.basename(images_path)} & {os.path.basename(labels_path)}...")
    with gzip.open(labels_path, "rb") as f:
        f.read(8)
        raw_labels = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(images_path, "rb") as f:
        f.read(16)
        raw_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    # Stratified balanced sampling across all 62 classes
    selected_indices = []
    for c in range(NUM_CLASSES):
        c_idxs = np.where(raw_labels == c)[0]
        if len(c_idxs) > max_per_class:
            selected_c = np.random.choice(c_idxs, max_per_class, replace=False)
        else:
            selected_c = c_idxs
        selected_indices.extend(selected_c)

    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)

    images = raw_images[selected_indices]
    labels = raw_labels[selected_indices]

    # EMNIST raw images are stored transposed; transpose each to make upright
    images = np.transpose(images, (0, 2, 1))

    # Normalize to [0.0, 1.0] and add channel dimension
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)

    print(f"  Loaded {len(images)} balanced samples across {NUM_CLASSES} classes (Shape: {images.shape})")
    return images, labels


def load_and_preprocess_data(data_dir="data/emnist", train_samples_per_class=2000, test_samples_per_class=300):
    """
    Load EMNIST 'byclass' dataset from local ubyte files or downloads directory.
    """
    if not os.path.exists(data_dir):
        alt_dir = r"C:\Users\K.Gnaneshwar\tensorflow_datasets\downloads\extracted\ZIP.biometrics.nist.gov_cs_links_EMNIST_gzip-5u2fjN3KpzAuJXk7PNtLPNb6LcJaTw1ZM6ioBn82o4.zip\gzip"
        if os.path.exists(alt_dir):
            data_dir = alt_dir

    train_img_p = os.path.join(data_dir, "emnist-byclass-train-images-idx3-ubyte.gz")
    train_lbl_p = os.path.join(data_dir, "emnist-byclass-train-labels-idx1-ubyte.gz")
    test_img_p = os.path.join(data_dir, "emnist-byclass-test-images-idx3-ubyte.gz")
    test_lbl_p = os.path.join(data_dir, "emnist-byclass-test-labels-idx1-ubyte.gz")

    x_train, y_train_raw = load_ubyte_images_and_labels(train_img_p, train_lbl_p, max_per_class=train_samples_per_class)
    x_test, y_test_raw = load_ubyte_images_and_labels(test_img_p, test_lbl_p, max_per_class=test_samples_per_class)

    y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

    return x_train, y_train, x_test, y_test, y_test_raw


def build_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    """
    Build a high-performance CNN architecture for 62 EMNIST categories:
    - Block 1: Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    - Block 2: Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    - Block 3: Conv2D(128) + BatchNorm + MaxPool + Dropout(0.3)
    - Dense(256) + BatchNorm + Dropout(0.4)
    - Dense(62, softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Conv Block 3
        layers.Conv2D(128, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def plot_training_history(history, save_path="emnist_training_history.png"):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    axes[0].set_title("EMNIST Training and Validation Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    axes[1].plot(history.history["loss"], label="Training Loss", marker="o")
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="Validation Loss", marker="o")
    axes[1].set_title("EMNIST Training and Validation Loss", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"\nTraining curves saved to '{save_path}'")


def evaluate_and_report(model, x_test, y_test, y_test_raw, target_names, confusion_matrix_path="emnist_confusion_matrix.png"):
    """Evaluate model on test set, print classification report, and plot confusion matrix."""
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON EMNIST TEST SET")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (62 Classes)")
    print("=" * 60)
    print(classification_report(y_test_raw, y_pred, target_names=target_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test_raw, y_pred, labels=list(range(NUM_CLASSES)))

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title("EMNIST ByClass Confusion Matrix (62 Classes)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(confusion_matrix_path)), exist_ok=True)
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix plot saved to '{confusion_matrix_path}'")


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    # 1. Print class count and label mapping
    mapping, target_names = get_emnist_byclass_mapping()

    # 2. Load and preprocess balanced EMNIST ByClass data (124,000 train + 18,600 test)
    x_train, y_train, x_test, y_test, y_test_raw = load_and_preprocess_data(
        data_dir="data/emnist",
        train_samples_per_class=2000,
        test_samples_per_class=300
    )

    # 3. Build CNN architecture for 62 classes
    model = build_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)
    model.summary()

    # 4. Compile model with Adam
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    # 6. Train model
    epochs = 8
    batch_size = 128
    validation_split = 0.1

    print("\n" + "=" * 60)
    print(f"TRAINING CNN ON REAL EMNIST BYCLASS ({epochs} Epochs, Batch Size: {batch_size})")
    print("=" * 60)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Save model
    model.save("emnist_model.keras")
    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "emnist_model.keras"))
    print("\nModel successfully saved as 'emnist_model.keras' and 'models/emnist_model.keras'")

    # 8. Plot training curves
    plot_training_history(history, save_path="emnist_training_history.png")
    plot_training_history(history, save_path="reports/emnist_training_history.png")

    # 9. Evaluate model
    evaluate_and_report(
        model,
        x_test,
        y_test,
        y_test_raw,
        target_names=target_names,
        confusion_matrix_path="emnist_confusion_matrix.png"
    )
    evaluate_and_report(
        model,
        x_test,
        y_test,
        y_test_raw,
        target_names=target_names,
        confusion_matrix_path="reports/emnist_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
