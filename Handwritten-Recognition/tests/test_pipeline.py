"""
Automated Test Suite for Handwritten Recognition System

Implements test cases defined in Section 10 of project specification:
- TC01: Preprocessing Grayscale & [0, 1] Normalization
- TC02: Output Tensor Shape (1, 28, 28, 1)
- TC03: Morphological Stroke Dilation
- TC04: Center of Mass Centering
- TC05: MNIST CNN Architecture & Compilation
- TC06: EMNIST CNN Architecture & 62 Classes
- TC07: Model File Loading & Cache Verification
- TC08: Top-1 and Top-3 Prediction Structure
- TC09: Label Mappings Integrity
- TC10: Inference Latency Performance (< 1000ms)
"""

import os
import time
import unittest
import numpy as np
from PIL import Image
import tensorflow as tf
from src.model import build_cnn_model, compile_model
from src.preprocess import preprocess_image
from src.utils import get_emnist_mapping, get_mnist_mapping


class TestHandwrittenRecognitionPipeline(unittest.TestCase):

    def setUp(self):
        # Create synthetic test canvas image (300x300, white background with black centered square/stroke)
        img_arr = np.ones((300, 300, 3), dtype=np.uint8) * 255
        # Draw a black vertical line simulating digit '1'
        img_arr[50:250, 140:160] = 0
        self.sample_img = Image.fromarray(img_arr)

    def test_tc01_preprocessing_normalization(self):
        """TC01: Verify preprocessed values are strictly normalized within [0.0, 1.0]."""
        tensor, _ = preprocess_image(self.sample_img)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertEqual(tensor.dtype, np.float32)

    def test_tc02_preprocessing_tensor_shape(self):
        """TC02: Verify preprocessed output tensor matches expected CNN input (1, 28, 28, 1)."""
        tensor, preview = preprocess_image(self.sample_img)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertEqual(preview.size, (28, 28))

    def test_tc03_stroke_dilation(self):
        """TC03: Verify morphological dilation thickens non-zero stroke pixels."""
        tensor_thin, _ = preprocess_image(self.sample_img, thicken_stroke=False)
        tensor_thick, _ = preprocess_image(self.sample_img, thicken_stroke=True)
        nonzero_thin = np.count_nonzero(tensor_thin > 0.1)
        nonzero_thick = np.count_nonzero(tensor_thick > 0.1)
        self.assertGreaterEqual(nonzero_thick, nonzero_thin)

    def test_tc04_center_of_mass(self):
        """TC04: Verify character is positioned at or near center of 28x28 grid."""
        tensor, _ = preprocess_image(self.sample_img)
        from scipy.ndimage import center_of_mass
        cy, cx = center_of_mass(tensor[0, :, :, 0])
        # Center of mass should be close to 14 (+/- 2 pixels)
        self.assertAlmostEqual(cx, 14.0, delta=2.5)
        self.assertAlmostEqual(cy, 14.0, delta=2.5)

    def test_tc05_mnist_model_architecture(self):
        """TC05: Verify MNIST model builds with 10 output classes and compiles properly."""
        model = build_cnn_model(num_classes=10)
        compile_model(model)
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(len(model.layers), 10)

    def test_tc06_emnist_model_architecture(self):
        """TC06: Verify EMNIST model builds with 62 output classes."""
        model = build_cnn_model(num_classes=62)
        compile_model(model)
        self.assertEqual(model.output_shape, (None, 62))

    def test_tc07_model_file_loading(self):
        """TC07: Verify saved .keras model files load successfully."""
        mnist_path = "models/mnist_model.keras" if os.path.exists("models/mnist_model.keras") else "mnist_model.keras"
        if os.path.exists(mnist_path):
            model = tf.keras.models.load_model(mnist_path)
            self.assertIsNotNone(model)
            self.assertEqual(model.output_shape, (None, 10))

    def test_tc08_prediction_format(self):
        """TC08: Verify model output produces valid probability distribution summing to 1."""
        model = build_cnn_model(num_classes=10)
        tensor, _ = preprocess_image(self.sample_img)
        probs = model.predict(tensor, verbose=0)[0]
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_tc09_label_mappings(self):
        """TC09: Verify MNIST and EMNIST label mappings completeness."""
        mnist_map = get_mnist_mapping()
        emnist_map = get_emnist_mapping()
        self.assertEqual(len(mnist_map), 10)
        self.assertEqual(len(emnist_map), 62)
        self.assertEqual(mnist_map[0], "0")
        self.assertEqual(emnist_map[10], "A")
        self.assertEqual(emnist_map[36], "a")

    def test_tc10_inference_latency(self):
        """TC10: Verify inference latency is strictly under 1000ms (1s)."""
        model = build_cnn_model(num_classes=10)
        tensor, _ = preprocess_image(self.sample_img)
        # Warmup
        _ = model.predict(tensor, verbose=0)
        # Measure
        start_time = time.time()
        _ = model.predict(tensor, verbose=0)
        latency_ms = (time.time() - start_time) * 1000
        print(f"\n[Performance] Inference Latency: {latency_ms:.2f} ms")
        self.assertLess(latency_ms, 1000.0)


if __name__ == "__main__":
    unittest.main()
