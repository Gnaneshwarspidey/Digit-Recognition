# ✍️ Handwritten Digit & Character Recognition

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning system for real-time handwritten digit and character classification using Convolutional Neural Networks (CNN) trained on **MNIST** (10 classes: `0-9`) and **EMNIST ByClass** (62 classes: `0-9`, `A-Z`, `a-z`), paired with an interactive **Streamlit** web application with drawing canvas and image upload support.

---

## 🚀 Live Demo

- **Streamlit Cloud App**: `[Add your deployed Streamlit App URL here]` *(e.g. https://handwritten-recognition.streamlit.app)*

---

## ✨ Features

- **Dual Model Support**:
  - **MNIST CNN**: Highly optimized for digit recognition (`0` through `9`) reaching **99.37% test accuracy**.
  - **EMNIST ByClass CNN**: Comprehensive 62-class recognition (Digits `0-9`, Uppercase `A-Z`, Lowercase `a-z`).
- **Interactive Web Interface**:
  - **Real-Time Drawing Canvas**: Draw digits or characters directly on the browser using `streamlit-drawable-canvas`.
  - **Image Upload**: Upload `.png`, `.jpg`, or `.jpeg` files.
  - **Clear & Brush Controls**: Adjustable stroke width and one-click canvas reset.
- **MNIST-Standard Preprocessing Engine**:
  - Alpha transparency flattening over solid backgrounds.
  - Automatic background inversion (bright foreground strokes on black background).
  - Otsu automated binarization thresholding for noise suppression.
  - Morphological dilation to ensure thin canvas strokes match dataset distribution.
  - Aspect-ratio preserving scaling into standard $20 \times 20$ inner bounding box.
  - Center-of-Mass translation to exactly $(14, 14)$ on the $28 \times 28$ grid.
- **Predictions & Visual Diagnostics**:
  - Top 1 prediction with confidence score.
  - Top 3 probability rankings with interactive confidence meters.
  - Expandable live visual inspection of the preprocessed $28 \times 28$ tensor.

---

## 📂 Project Structure

```text
Handwritten-Recognition/
├── mnist_model.keras             # Trained MNIST CNN model weights (~99.37% acc)
├── emnist_model.keras            # Trained EMNIST ByClass CNN model weights
├── mnist_train.py                # MNIST model training & evaluation script
├── emnist_train.py               # EMNIST ByClass model training script
├── streamlit_app.py              # Interactive Streamlit application
├── mnist_confusion_matrix.png    # MNIST confusion matrix evaluation plot
├── mnist_training_history.png    # MNIST training accuracy & loss curves
├── emnist_confusion_matrix.png   # EMNIST confusion matrix evaluation plot
├── emnist_training_history.png   # EMNIST training accuracy & loss curves
├── requirements.txt              # Project package dependencies
└── README.md                     # Project documentation
```

---

## 🧠 Model Architecture

Both models utilize a high-performing Convolutional Neural Network (CNN) architecture:

```
Input (28, 28, 1)
  │
  ├── Conv2D (32 filters, 3x3, ReLU, Same Padding)
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (64 filters, 3x3, ReLU, Same Padding)
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (128 filters, 3x3, ReLU, Same Padding)
  ├── MaxPooling2D (2x2)
  │
  ├── Flatten
  ├── Dense (128 units, ReLU)
  ├── Dropout (0.5)
  │
  └── Dense (10 units for MNIST / 62 units for EMNIST, Softmax)
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Handwritten-Recognition.git
cd Handwritten-Recognition
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Models

### Train MNIST (Digits 0-9)
```bash
python mnist_train.py
```
*Generates `mnist_model.keras`, `mnist_training_history.png`, and `mnist_confusion_matrix.png`.*

### Train EMNIST ByClass (Digits + Letters)
```bash
python emnist_train.py
```
*Generates `emnist_model.keras`, `emnist_training_history.png`, and `emnist_confusion_matrix.png`.*

---

## 🖥️ Running the Web Application

Launch the Streamlit interactive UI locally:

```bash
python -m streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Preprocessing Pipeline

To guarantee high prediction accuracy on freehand drawings:

```
[Raw Input Image / Canvas]
         │
         ▼
[1. Convert Grayscale + Alpha Flattening]
         │
         ▼
[2. Invert Brightness if Light BG]
         │
         ▼
[3. Otsu's Adaptive Thresholding]
         │
         ▼
[4. Morphological Dilation (Stroke Thickening)]
         │
         ▼
[5. Non-Zero Bounding Box Cropping]
         │
         ▼
[6. Aspect-Preserved Resize to 20x20]
         │
         ▼
[7. Embed into 28x28 Black Canvas]
         │
         ▼
[8. Shift Center-of-Mass to (14, 14)]
         │
         ▼
[9. Normalize to [0.0, 1.0] -> Reshape (1, 28, 28, 1)]
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Handwritten recognition system"
   git remote add origin https://github.com/<your-username>/Handwritten-Recognition.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository.
4. Set Main file path to `streamlit_app.py` and click **Deploy**.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
