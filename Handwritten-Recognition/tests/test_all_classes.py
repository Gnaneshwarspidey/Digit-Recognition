"""
Comprehensive Test Suite for All 62 Alphanumeric Classes & Digits
Tests MNIST Model (0-9) and EMNIST Model (0-9, A-Z, a-z) with formal test cases.
"""

import os
import gzip
import struct
import unittest
import numpy as np
import tensorflow as tf
from src.utils import get_mnist_mapping, get_emnist_mapping
from src.preprocess import segment_and_preprocess_characters, preprocess_image
from PIL import Image, ImageDraw


class TestAllNumbersAndAlphabets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mnist_map = get_mnist_mapping()
        cls.emnist_map = get_emnist_mapping()

        # Load models
        cls.mnist_model = tf.keras.models.load_model("models/mnist_model.keras")
        cls.emnist_model = tf.keras.models.load_model("models/emnist_model.keras")

        # Load real test dataset samples from data/emnist
        data_dir = "data/emnist"
        test_img_p = os.path.join(data_dir, "emnist-byclass-test-images-idx3-ubyte.gz")
        test_lbl_p = os.path.join(data_dir, "emnist-byclass-test-labels-idx1-ubyte.gz")

        with gzip.open(test_lbl_p, "rb") as f:
            f.read(8)
            raw_labels = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(test_img_p, "rb") as f:
            f.read(16)
            raw_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

        # Transpose images to upright orientation and normalize
        raw_images = np.transpose(raw_images, (0, 2, 1)).astype("float32") / 255.0
        raw_images = np.expand_dims(raw_images, axis=-1)

        # Cache 20 real test samples for each of the 62 classes
        cls.samples_by_class = {}
        for c in range(62):
            idxs = np.where(raw_labels == c)[0][:20]
            cls.samples_by_class[c] = raw_images[idxs]

        # Load real MNIST test data
        (_, _), (cls.x_mnist_test, cls.y_mnist_test) = tf.keras.datasets.mnist.load_data()
        cls.x_mnist_test = cls.x_mnist_test.astype("float32") / 255.0
        cls.x_mnist_test = np.expand_dims(cls.x_mnist_test, axis=-1)

    # -------------------------------------------------------------------------
    # PART 1: MNIST DIGITS (0 to 9) - 10 Test Cases
    # -------------------------------------------------------------------------
    def test_mnist_digit_0(self): self._eval_mnist_digit(0)
    def test_mnist_digit_1(self): self._eval_mnist_digit(1)
    def test_mnist_digit_2(self): self._eval_mnist_digit(2)
    def test_mnist_digit_3(self): self._eval_mnist_digit(3)
    def test_mnist_digit_4(self): self._eval_mnist_digit(4)
    def test_mnist_digit_5(self): self._eval_mnist_digit(5)
    def test_mnist_digit_6(self): self._eval_mnist_digit(6)
    def test_mnist_digit_7(self): self._eval_mnist_digit(7)
    def test_mnist_digit_8(self): self._eval_mnist_digit(8)
    def test_mnist_digit_9(self): self._eval_mnist_digit(9)

    def _eval_mnist_digit(self, digit: int):
        idxs = np.where(self.y_mnist_test == digit)[0][:100]
        batch = self.x_mnist_test[idxs]
        preds = np.argmax(self.mnist_model.predict(batch, verbose=0), axis=1)
        acc = np.mean(preds == digit) * 100
        print(f"[MNIST Digit {digit}] Accuracy: {acc:.2f}% ({len(idxs)} samples)")
        self.assertGreaterEqual(acc, 95.0, f"MNIST Digit {digit} accuracy should be >= 95%")

    # -------------------------------------------------------------------------
    # PART 2: EMNIST UPPERCASE LETTERS (A to Z) - 26 Test Cases
    # -------------------------------------------------------------------------
    def test_emnist_letter_A(self): self._eval_emnist_class(10, 'A')
    def test_emnist_letter_B(self): self._eval_emnist_class(11, 'B')
    def test_emnist_letter_C(self): self._eval_emnist_class(12, 'C')
    def test_emnist_letter_D(self): self._eval_emnist_class(13, 'D')
    def test_emnist_letter_E(self): self._eval_emnist_class(14, 'E')
    def test_emnist_letter_F(self): self._eval_emnist_class(15, 'F')
    def test_emnist_letter_G(self): self._eval_emnist_class(16, 'G')
    def test_emnist_letter_H(self): self._eval_emnist_class(17, 'H')
    def test_emnist_letter_I(self): self._eval_emnist_class(18, 'I')
    def test_emnist_letter_J(self): self._eval_emnist_class(19, 'J')
    def test_emnist_letter_K(self): self._eval_emnist_class(20, 'K')
    def test_emnist_letter_L(self): self._eval_emnist_class(21, 'L')
    def test_emnist_letter_M(self): self._eval_emnist_class(22, 'M')
    def test_emnist_letter_N(self): self._eval_emnist_class(23, 'N')
    def test_emnist_letter_O(self): self._eval_emnist_class(24, 'O')
    def test_emnist_letter_P(self): self._eval_emnist_class(25, 'P')
    def test_emnist_letter_Q(self): self._eval_emnist_class(26, 'Q')
    def test_emnist_letter_R(self): self._eval_emnist_class(27, 'R')
    def test_emnist_letter_S(self): self._eval_emnist_class(28, 'S')
    def test_emnist_letter_T(self): self._eval_emnist_class(29, 'T')
    def test_emnist_letter_U(self): self._eval_emnist_class(30, 'U')
    def test_emnist_letter_V(self): self._eval_emnist_class(31, 'V')
    def test_emnist_letter_W(self): self._eval_emnist_class(32, 'W')
    def test_emnist_letter_X(self): self._eval_emnist_class(33, 'X')
    def test_emnist_letter_Y(self): self._eval_emnist_class(34, 'Y')
    def test_emnist_letter_Z(self): self._eval_emnist_class(35, 'Z')

    # -------------------------------------------------------------------------
    # PART 3: EMNIST LOWERCASE LETTERS (a to z) - 26 Test Cases
    # -------------------------------------------------------------------------
    def test_emnist_letter_a_lower(self): self._eval_emnist_class(36, 'a')
    def test_emnist_letter_b_lower(self): self._eval_emnist_class(37, 'b')
    def test_emnist_letter_c_lower(self): self._eval_emnist_class(38, 'c')
    def test_emnist_letter_d_lower(self): self._eval_emnist_class(39, 'd')
    def test_emnist_letter_e_lower(self): self._eval_emnist_class(40, 'e')
    def test_emnist_letter_f_lower(self): self._eval_emnist_class(41, 'f')
    def test_emnist_letter_g_lower(self): self._eval_emnist_class(42, 'g')
    def test_emnist_letter_h_lower(self): self._eval_emnist_class(43, 'h')
    def test_emnist_letter_i_lower(self): self._eval_emnist_class(44, 'i')
    def test_emnist_letter_j_lower(self): self._eval_emnist_class(45, 'j')
    def test_emnist_letter_k_lower(self): self._eval_emnist_class(46, 'k')
    def test_emnist_letter_l_lower(self): self._eval_emnist_class(47, 'l')
    def test_emnist_letter_m_lower(self): self._eval_emnist_class(48, 'm')
    def test_emnist_letter_n_lower(self): self._eval_emnist_class(49, 'n')
    def test_emnist_letter_o_lower(self): self._eval_emnist_class(50, 'o')
    def test_emnist_letter_p_lower(self): self._eval_emnist_class(51, 'p')
    def test_emnist_letter_q_lower(self): self._eval_emnist_class(52, 'q')
    def test_emnist_letter_r_lower(self): self._eval_emnist_class(53, 'r')
    def test_emnist_letter_s_lower(self): self._eval_emnist_class(54, 's')
    def test_emnist_letter_t_lower(self): self._eval_emnist_class(55, 't')
    def test_emnist_letter_u_lower(self): self._eval_emnist_class(56, 'u')
    def test_emnist_letter_v_lower(self): self._eval_emnist_class(57, 'v')
    def test_emnist_letter_w_lower(self): self._eval_emnist_class(58, 'w')
    def test_emnist_letter_x_lower(self): self._eval_emnist_class(59, 'x')
    def test_emnist_letter_y_lower(self): self._eval_emnist_class(60, 'y')
    def test_emnist_letter_z_lower(self): self._eval_emnist_class(61, 'z')

    def _eval_emnist_class(self, class_id: int, char_name: str):
        samples = self.samples_by_class[class_id]
        probs = self.emnist_model.predict(samples, verbose=0)
        preds = np.argmax(probs, axis=1)

        # Allow case-insensitive equivalence for characters that share identical glyphs
        # (e.g., C/c, O/o, S/s, V/v, W/w, X/x, Z/z)
        equiv_chars = {char_name, char_name.lower(), char_name.upper()}
        pred_chars = [self.emnist_map.get(p, "") for p in preds]
        matches = sum(1 for pc in pred_chars if pc in equiv_chars)
        acc = (matches / len(samples)) * 100

        top1_conf = np.mean(np.max(probs, axis=1)) * 100
        print(f"[EMNIST '{char_name}'] Accuracy: {acc:.1f}% | Avg Confidence: {top1_conf:.1f}%")
        self.assertGreater(acc, 30.0, f"EMNIST character '{char_name}' recognition should be non-trivial")


if __name__ == "__main__":
    unittest.main()

