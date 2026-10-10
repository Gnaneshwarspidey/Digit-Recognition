"""
Command Line Prediction Tool for Handwritten Recognition

Usage:
    python predict.py --model mnist --image sample.png
    python predict.py --model emnist --image sample_char.png
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from src.preprocess import preprocess_image
from src.utils import get_emnist_mapping, get_mnist_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Predict handwritten digit or character from image")
    parser.add_argument(
        "--model",
        type=str,
        default="mnist",
        choices=["mnist", "emnist"],
        help="Model type: 'mnist' (0-9) or 'emnist' (0-9, A-Z, a-z)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to model .keras weights file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file (.png, .jpg, .jpeg)"
    )
    parser.add_argument(
        "--no-thicken",
        action="store_true",
        help="Disable morphological stroke dilation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        sys.exit(1)

    # Determine weights path
    if args.weights is not None:
        model_path = args.weights
    else:
        model_name = f"{args.model}_model.keras"
        # Check current dir or models/ dir
        if os.path.exists(model_name):
            model_path = model_name
        elif os.path.exists(os.path.join("models", model_name)):
            model_path = os.path.join("models", model_name)
        else:
            print(f"Error: Model weights '{model_name}' not found. Please train it first.")
            sys.exit(1)

    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # Select mapping
    label_map = get_mnist_mapping() if args.model == "mnist" else get_emnist_mapping()

    # Preprocess
    print(f"Preprocessing input image '{args.image}'...")
    tensor, _ = preprocess_image(args.image, thicken_stroke=not args.no_thicken)

    # Inference
    print("Running inference...")
    probs = model.predict(tensor, verbose=0)[0]

    # Results
    top1_idx = int(np.argmax(probs))
    top1_char = label_map.get(top1_idx, str(top1_idx))
    top1_conf = float(probs[top1_idx]) * 100

    print("\n" + "=" * 45)
    print("PREDICTION RESULT")
    print("=" * 45)
    print(f"Predicted Class: '{top1_char}'")
    print(f"Confidence:      {top1_conf:.2f}%")

    print("\nTop 3 Predictions:")
    top3_indices = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3_indices, start=1):
        char = label_map.get(int(idx), str(idx))
        conf = float(probs[idx]) * 100
        print(f"  #{rank}: '{char}' ({conf:.2f}%)")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()

