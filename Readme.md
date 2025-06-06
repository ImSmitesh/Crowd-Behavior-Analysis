# 🧠 Crowd Behavior Analysis – Violence Detection in Large Gatherings

![crowd-behavior](https://img.shields.io/badge/deep%20learning-CNN%20%2B%20LSTM-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-Bachelor's%20Project-success)

## 📌 Project Overview

This project focuses on **automated crowd behavior analysis**, particularly detecting **violent or aggressive behavior** in large public gatherings using video input. The model is trained to distinguish between *fight* and *non-fight* scenarios based on spatio-temporal patterns extracted from video data.

It was developed using **deep learning**, **MobileNetV2**, and **Bidirectional LSTMs**, and is trained on the **RWF-2000** dataset. 

---

## 🎯 Features

- Uses **MobileNetV2** for spatial feature extraction (lightweight CNN).
- Employs **Bidirectional LSTMs** to learn temporal dependencies in video sequences.
- Implements **custom data augmentation** to enhance model robustness.
- Includes **video data loader** with frame skipping and augmentation.
- Generates training and validation accuracy/loss graphs.
- Supports **model checkpointing** and saving best-performing models.

---

## 🧪 Dataset

**RWF-2000**: A video dataset consisting of 2,000 clips equally divided between violent (fight) and non-violent (non-fight) scenes, extracted from surveillance footage. (https://www.kaggle.com/datasets/vulamnguyen/rwf2000)

- Dataset location assumed to be: `./RWF-2000/train/...` and `./RWF-2000/val/...`
- Each video is preprocessed by selecting 30 frames (skipping every 5) and resizing them to `224x224`.

---

## 🚀 Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Temporal Modeling**: TimeDistributed + Bidirectional LSTM
- **Classification Head**: Fully connected layers ending in sigmoid activation

```plaintext
Input (30 frames of size 224x224)
↓
TimeDistributed(MobileNetV2)
↓
TimeDistributed(GlobalAvgPool2D)
↓
Bidirectional LSTM
↓
Dense → ReLU
↓
Dropout
↓
Dense → Sigmoid
