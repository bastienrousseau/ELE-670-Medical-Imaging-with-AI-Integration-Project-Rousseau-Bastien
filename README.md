# ELE-670-Medical-Imaging-with-AI-Integration-Project-Rousseau-Bastien

# 🩺 Breast Cancer Classification using Deep Learning  
**Author:** Bastien Rousseau  
**Email:** barou1817@uis.no  
**Course:** ELE670 – Medical Imaging and AI Integration  
**Institution:** University of Stavanger / Grenoble INP – Phelma  

---

## 📘 Project Overview
This project focuses on the **automatic classification of breast tumors** from mammography images into three categories:
- **Benign**
- **Malignant**
- **Benign without callback** (clearly benign lesions that do not require further examination)

The goal is to compare several **deep learning architectures** – from classical CNNs to modern Vision Transformers (ViT) – and evaluate their ability to distinguish between lesion types despite the small dataset size.

---

## 🧠 Methods

### 1. **Data**
The dataset used is the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** [Lee et al., 2017], derived from the historical DDSM collection.  
To make the images directly exploitable with deep learning frameworks, DICOM files were converted into **8-bit JPEGs**, with intensity normalization and metadata extraction.

### 2. **Preprocessing**
- Contrast and brightness normalization  
- Resizing (512×512 → 224×224 / 384×384 for ViT)  
- Augmentation: rotations, flips, Gaussian blur, sharpness, and noise  
- Dataset split: **80% training (with validation), 20% testing**

### 3. **Models**
| Model | Description | Key Notes |
|:------|:-------------|:-----------|
| **CNN (baseline)** | Small custom convolutional network | Overfitted due to small dataset |
| **ResNet-B0** | Transfer learning from ImageNet | Best accuracy (74% validation) |
| **Vision Transformer (ViT-B/16)** | Fine-tuned transformer with self-attention | Better AUC but sensitive to ImageNet bias |

### 4. **Loss & Training**
- **Focal Loss** to handle class imbalance  
- **CosineAnnealingLR** scheduler  
- **WeightedRandomSampler** for balanced batches  
- **Two-stage fine-tuning**: head → last 3 transformer blocks

---

## 📊 Results Summary

| Model | Val. Accuracy | Test Accuracy | AUC (macro) |
|:------|:--------------|:---------------|:-------------|
| CNN | ~55% | ~50% | 0.72 |
| ResNet-B0 | **74%** | 57% | 0.79 |
| ViT-B/16 | 58% | 54%| **0.86 (best class)** |

- The **ResNet** outperformed the CNN baseline thanks to transfer learning.  
- The **ViT** captured global context but required more data and domain adaptation.  
- Overfitting remains the main limitation due to dataset size.

---

## 📈 Visualization
Training and validation losses and accuracies are displayed per epoch using `matplotlib`.  
Example plots are automatically generated at the end of training.

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/breast-cancer-classifier.git
cd breast-cancer-classifier
