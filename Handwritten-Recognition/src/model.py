"""
CNN Architecture Module for MNIST & EMNIST Recognition
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10) -> models.Sequential:
    """
    Builds a standard 3-block Convolutional Neural Network:
    - Conv Block 1: Conv2D(32, 3x3, relu, padding='same') -> MaxPooling2D(2, 2)
    - Conv Block 2: Conv2D(64, 3x3, relu, padding='same') -> MaxPooling2D(2, 2)
    - Conv Block 3: Conv2D(128, 3x3, relu, padding='same') -> MaxPooling2D(2, 2)
    - Classifier:   Flatten -> Dense(128, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # Conv Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # Conv Block 3
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # Fully Connected Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


def compile_model(model: models.Sequential, learning_rate=0.001) -> models.Sequential:
    """Compiles the CNN model with Adam optimizer and categorical cross-entropy."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

