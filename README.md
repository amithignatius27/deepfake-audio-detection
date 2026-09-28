# Deepfake Audio Detection System

A deep learning-based system for detecting whether an audio sample is **real or AI-generated** using Log-Mel Spectrograms and a modified ResNet18 convolutional neural network.

The project implements an end-to-end pipeline covering audio preprocessing, feature extraction, model training, validation, fine-tuning with additional voice recordings, evaluation, and inference through a Flask-based web interface.

---

## Overview

AI-generated speech has become increasingly difficult to distinguish from authentic human speech. This project explores an audio-based approach to identify synthetic or manipulated speech by learning patterns from time-frequency representations of audio.

The system converts an input audio signal into a Log-Mel Spectrogram and uses a ResNet18-based classifier to predict whether the recording is:

* **REAL**
* **FAKE**

The model achieved approximately **95% validation accuracy** during development.

---

## System Pipeline

```text
Audio Input
     │
     ▼
Resample to 16 kHz
     │
     ▼
Mel-Spectrogram
128 Mel Bands
     │
     ▼
Log Scaling
     │
     ▼
Pad / Truncate
128 × 400
     │
     ▼
Normalization
     │
     ▼
Modified ResNet18
     │
     ▼
Binary Classification
     │
     ├──────────► REAL
     │
     └──────────► FAKE
```

---

## Key Features

* Audio classification using PyTorch
* Log-Mel Spectrogram-based feature extraction
* Modified ResNet18 architecture for single-channel spectrogram input
* 16 kHz audio standardization
* 128 Mel-frequency bands
* Fixed-size spectrogram representation
* Weighted Random Sampling for class imbalance
* Validation-based model checkpointing
* Fine-tuning using additional real-world voice recordings
* Standalone inference pipeline
* Flask-based web interface
* Softmax-based prediction confidence

---

## Dataset

The dataset is organized into separate real and fake audio samples for training, development, and evaluation.

```text
data/
├── train/
│   ├── real/
│   └── fake/
│
├── dev/
│   ├── real/
│   └── fake/
│
└── eval/
    ├── real/
    └── fake/
```

The dataset itself is not included in this repository.

A custom PyTorch dataset implementation loads audio files dynamically during training and inference.

---

## Audio Preprocessing

Before being passed to the neural network, each audio file goes through the following preprocessing steps.

### 1. Resampling

Audio is standardized to:

```text
Sampling Rate: 16,000 Hz
```

This provides a consistent sampling rate across the dataset.

### 2. Mel-Spectrogram

The waveform is converted into a Mel-Spectrogram using Librosa.

The main parameters are:

| Parameter     |  Value |
| ------------- | -----: |
| Sampling rate | 16 kHz |
| Mel bands     |    128 |
| FFT size      |    512 |
| Hop length    |    160 |
| Window length |    400 |

### 3. Log Scaling

The Mel-Spectrogram is converted to logarithmic scale:

```python
mel = np.log1p(mel)
```

This reduces the dynamic range of the spectral representation.

### 4. Fixed-Length Representation

Since audio recordings can have different durations, the spectrogram is converted to a fixed size:

```text
128 × 400
```

Shorter samples are padded and longer samples are truncated.

### 5. Normalization

The spectrogram is normalized before being passed to the model:

```python
mel = (mel - mel.mean()) / (mel.std() + 1e-6)
```

The final model input has the shape:

```text
[1, 128, 400]
```

---

# Model Architecture

The project uses **ResNet18** as the main feature extraction and classification architecture.

ResNet18 was originally designed for RGB images, so the first convolutional layer was modified to accept a **single-channel Mel-Spectrogram**.

### Input

```text
[1, 128, 400]
```

### Modified ResNet18

```text
Single-channel Mel-Spectrogram
            │
            ▼
       ResNet18 Backbone
            │
            ▼
      Feature Extraction
            │
            ▼
       Custom Classifier
            │
            ├── 512 → 256
            ├── BatchNorm
            ├── ReLU
            ├── Dropout
            ├── 256 → 64
            ├── ReLU
            └── 64 → 2
                    │
             ┌──────┴──────┐
             ▼             ▼
           REAL           FAKE
```

The final classification layer produces two outputs corresponding to the two classes.

The model implementation is available in:

```text
src/model.py
```

---

# Training

The model is trained using supervised learning with labeled real and fake audio samples.

### Training configuration

| Parameter             |                 Value |
| --------------------- | --------------------: |
| Model                 |              ResNet18 |
| Epochs                |                    20 |
| Batch size            |                    16 |
| Initial learning rate |                  1e-4 |
| Optimizer             |                 AdamW |
| Loss function         |         Cross Entropy |
| Scheduler             |     ReduceLROnPlateau |
| Sampling              | WeightedRandomSampler |

### Class imbalance

A `WeightedRandomSampler` is used during training to reduce the effect of class imbalance.

Instead of allowing the model to repeatedly see a larger class, sampling weights are calculated based on class frequency.

---

# Validation and Checkpointing

After each training epoch, the model is evaluated on the development dataset.

Validation accuracy is monitored throughout training.

When the model achieves a new best validation accuracy, its weights are saved as the best checkpoint.

```text
checkpoints/
└── best_model1.pth
```

The checkpoint directory is excluded from Git because trained model files can be large.

---

# Results

During development, the model achieved approximately:

| Metric              |   Result |
| ------------------- | -------: |
| Validation Accuracy | **~95%** |

The reported accuracy refers to the validation/development dataset used during model development.

Performance can vary depending on the dataset, recording conditions, audio quality, and type of synthetic speech.

---

# Fine-Tuning with Real-World Voice Recordings

To investigate how the trained model performs on additional recordings, a separate fine-tuning pipeline was implemented.

The fine-tuning process consists of two stages.

### Stage 1 — Classifier Fine-Tuning

The ResNet backbone is frozen while the classification layers are trained using additional audio samples.

### Stage 2 — Partial Backbone Fine-Tuning

Selected deeper layers of the ResNet backbone are unfrozen along with the classifier, allowing the model to adapt to the additional recordings.

The fine-tuning implementation is available in:

```text
src/finetune.py
```

A stereo-specific implementation is also included:

```text
src/finetune_stereo.py
```

---

# Inference

The standalone inference pipeline is implemented in:

```text
src/infer.py
```

The same preprocessing pipeline used during training is applied to an external audio file.

```text
External Audio
      │
      ▼
Preprocessing
      │
      ▼
Log-Mel Spectrogram
      │
      ▼
ResNet18
      │
      ▼
Softmax
      │
      ▼
Prediction + Confidence
```

Example:

```text
Prediction : FAKE
Confidence : 94.32%
```

The confidence value is derived from the model's Softmax probabilities.

---

# Web Application

A simple Flask web application is included to make the model easier to test.

### Backend

```text
src/app.py
```

### Frontend

```text
templates/index.html
```

The application allows a user to upload an audio file and receive a prediction from the trained model.

The backend exposes:

```text
GET  /
POST /predict
```

The `/predict` endpoint receives the uploaded audio, runs the inference pipeline, and returns the predicted class and confidence.

---

# Project Structure

```text
deepfake-audio-detection/
│
├── src/
│   ├── app.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── dataset_stereo.py
│   ├── eval_folder.py
│   ├── evaluate.py
│   ├── finetune.py
│   ├── finetune_stereo.py
│   ├── infer.py
│   ├── model.py
│   └── train.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/amithignatius27/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train the model

```bash
python src/train.py
```

## Evaluate the model

```bash
python src/evaluate.py
```

## Fine-tune the model

```bash
python src/finetune.py
```

## Run inference

```bash
python src/infer.py
```

## Start the web application

```bash
python src/app.py
```

---

# Limitations

* The dataset is not included in the repository.
* Trained model checkpoints are not stored in GitHub.
* The reported ~95% accuracy is based on the validation/development setup used during development.
* Performance may vary on audio generated using techniques that differ from those represented in the training data.
* Background noise, compression, recording devices, and other audio conditions can affect predictions.
* A model prediction should not be treated as definitive proof that an audio recording is authentic or manipulated.

---

# Future Improvements

Some areas that could be explored further:

* Evaluate on additional deepfake audio datasets.
* Add Precision, Recall, F1-score, ROC-AUC and EER.
* Test cross-dataset generalization.
* Add more noise and compression augmentation.
* Compare ResNet18 with audio-specific architectures.
* Improve the web interface.
* Containerize the application with Docker.
* Add automated testing and CI/CD.
* Deploy the inference service.

---

# Technologies

* Python
* PyTorch
* Torchvision
* Librosa
* NumPy
* Scikit-learn
* SoundFile
* Flask
* Flask-CORS
* HTML
* JavaScript
* Git
* GitHub

---

# Author

**Amith Ignatius **

GitHub: [amithignatius27](https://github.com/amithignatius27)

---

## Project

**Deepfake Audio Detection System — 2025**

Built as an applied deep learning project to explore detection of AI-generated speech using audio signal processing and convolutional neural networks.
