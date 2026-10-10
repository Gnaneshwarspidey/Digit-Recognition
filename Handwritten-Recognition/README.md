# Handwritten Digit and Character Recognition

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning system for real-time handwritten digit and alphanumeric character classification using Convolutional Neural Networks (CNN) trained on **MNIST** (10 classes: 0-9) and **EMNIST ByClass** (62 classes: 0-9, A-Z, a-z), paired with an interactive **Streamlit** web application featuring a skeuomorphic stationery canvas interface.

---

## Benchmarks and Performance Metrics

| Metric / Engine | MNIST Digits Engine | EMNIST Alphanumeric Engine |
| :--- | :--- | :--- |
| Target Classes | 10 (0 through 9) | 62 (0-9, A-Z, a-z) |
| Test Accuracy | **99.37%** (10,000 test digits) | **78.10%** (Top-3 Accuracy: **94.6%**) |
| Inference Latency | **< 5 ms** (Real-time CPU) | **< 8 ms** (Real-time CPU) |
| Model Architecture | Dual Conv2D + BatchNorm + Dropout | Deep Conv2D + BatchNorm + Dropout |
| Parameters | 1,199,882 trainable parameters | 1,226,558 trainable parameters |
| Test Suite Coverage | **72 / 72 passing tests** (100%) | Full 62-class automated verification |

---

## Features

- **Interactive User Interface**:
  - Deckled edge watercolor sketchpad canvas with zero-latency fluid freehand drawing.
  - Tactile controls for Undo, Redo, Clear, and Recognize operations.
  - Letterpress cardstock prediction display with live confidence scores.
  - Dedicated sidebar for model engine selection (MNIST vs EMNIST) and specifications.

- **Dual Neural Engines**:
  - **MNIST CNN**: Highly optimized for digit recognition (0-9) achieving 99.37% benchmark accuracy.
  - **EMNIST ByClass CNN**: Comprehensive 62-class alphanumeric recognition with orientation correction.

- **Preprocessing Pipeline (MNIST Standard)**:
  - Alpha transparency flattening over solid backgrounds.
  - Automatic background inversion (bright foreground strokes on dark background).
  - Otsu automated binarization thresholding for noise suppression.
  - Morphological dilation to ensure thin canvas strokes match training distribution.
  - Aspect-ratio preserving scaling into standard 20x20 inner bounding box.
  - Center-of-Mass translation to (14, 14) on a 28x28 grid.

- **Diagnostics and Metrics**:
  - Confusion matrix heatmaps and training convergence loss/accuracy curves.
  - Comprehensive documentation and diagnostic reports.

---

## Project Structure

```text
Handwritten-Recognition/
├── assets/                   # Sample test digit and letter images
├── models/                   # Pretrained high-accuracy Keras CNN models
│   ├── mnist_model.keras     # MNIST model (99.37% accuracy)
│   └── emnist_model.keras    # EMNIST 62-class model (78.1% accuracy)
├── reports/                  # Confusion matrices and training history plots
│   ├── mnist_confusion_matrix.png
│   ├── mnist_training_history.png
│   ├── emnist_confusion_matrix.png
│   └── emnist_training_history.png
├── src/                      # Modular production source code
│   ├── preprocess.py         # 9-step Center-of-Mass and segmentation pipeline
│   ├── model.py              # Modular CNN architecture builder
│   └── utils.py              # EMNIST/MNIST mapping and rendering utilities
├── tests/                    # Automated unit and E2E test suites (72/72 passing)
│   ├── test_pipeline.py      # Preprocessing and pipeline tests
│   └── test_all_classes.py   # Full 62-class prediction verification
├── streamlit_app.py          # Interactive skeuomorphic web application
├── app.py                    # Entrypoint alias
├── predict.py                # Standalone CLI inference tool
├── evaluate.py               # Standalone evaluation and reporting tool
├── mnist_train.py            # MNIST training script
├── emnist_train.py           # EMNIST training script
├── requirements.txt          # Pinned Python package dependencies
├── LICENSE                   # MIT License
└── README.md                 # Technical documentation
```

---

## Model Architecture

```text
Input (28, 28, 1)
  │
  ├── Conv2D (32 filters, 3x3, ReLU, Same Padding)
  ├── BatchNormalization
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (64 filters, 3x3, ReLU, Same Padding)
  ├── BatchNormalization
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (128 filters, 3x3, ReLU, Same Padding)
  ├── BatchNormalization
  ├── MaxPooling2D (2x2)
  │
  ├── Flatten
  ├── Dense (128 units, ReLU)
  ├── Dropout (0.5)
  │
  └── Dense (10 units for MNIST / 62 units for EMNIST, Softmax)
```

---

## Quick Start and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Gnaneshwarspidey/Digit-Recognition.git
cd Digit-Recognition/Handwritten-Recognition
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python -m streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501` or `http://localhost:8504`.

---

## Running Automated Tests

Run the complete test suite (72 unit and pipeline tests):

```bash
python -m pytest tests/ -v
```

---

## Training Models from Scratch

### Train MNIST (Digits 0-9):
```bash
python mnist_train.py
```

### Train EMNIST ByClass (Alphanumeric 62 classes):
```bash
python emnist_train.py
```

---

## Deployment on Streamlit Cloud

1. Fork or push the repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository.
4. Set the Main file path to `Handwritten-Recognition/streamlit_app.py`.
5. Deploy.

---

## License

This project is licensed under the [MIT License](LICENSE).
