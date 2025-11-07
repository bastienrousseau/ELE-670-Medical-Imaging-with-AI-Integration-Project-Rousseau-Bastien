# ELE-670-Medical-Imaging-with-AI-Integration-Project-Rousseau-Bastien

# Breast Cancer Classification 
**Author:** Bastien Rousseau  
**Email:** barou1817@uis.no  
**Course:** ELE670 – Medical Imaging and AI Integration  
**Institution:** University of Stavanger 

---

## Project Overview
This project focuses on the **automatic classification of breast tumors** from mammography images into three categories:
- **Benign**
- **Malignant**
- **Benign without callback** (clearly benign lesions that do not require further examination)

The goal is to compare several **deep learning architectures** – from ResNet model to modern Vision Transformers (ViT).

---

## Methods

### 1. **Preprocessing**
- Contrast and brightness normalization  
- Resizing (512×512 → 224×224 / 384×384 for ViT)  
- Augmentation: rotations, flips, Gaussian blur, sharpness, and noise  
- Dataset split: **80% training (with validation), 20% testing**
- Merging of class 

### 3. **Models**
| Model | Description | Key Notes |
|:------|:-------------|:-----------|
| **ResNet-B1** | Transfer learning from ImageNet | Best accuracy (74% validation) |
| **Vision Transformer (ViT-B/16)** | Fine-tuned transformer with self-attention | Better AUC but sensitive to ImageNet bias |

### 4. **Loss & Training**
- **Focal Loss** to handle class imbalance  
- **CosineAnnealingLR** scheduler  
- **WeightedRandomSampler** for balanced batches  
- **Two-stage fine-tuning**: head → last 3 transformer blocks

---

##  Results Summary

| Model | Val. Accuracy | Test Accuracy | AUC (macro) |
|:------|:--------------|:---------------|:-------------|
| ResNet-B0 | **74%** | 57% | 0.79 |
| ViT-B/16 | 58% | 54%| **0.86 (for the best class)** |

- The **ResNet** outperformed the firstly CNN baseline (which is not included in the code anymore) thanks to transfer learning.  
- The **ViT** captured global context but required more data and domain adaptation but showed good result about malignant class.  
- Overfitting remains the main limitation due to small dataset and the strong pre-training for the ViT.

---

##  Visualization
Training and validation losses and accuracies are displayed per epoch using `matplotlib`.  
Plots are automatically generated at the end of training.

---

## How to Run

### Download the notebook

Dowload the notebook file : project-ele670-breast-cancer-detection-Rousseau-Bastien.ipynb
You can then go to the folder where you downloaded the project and import it on the website kaggle after creating your account: https://www.kaggle.com/
Once done, import the notebook and add to the input section the dataset CBIS-DDSM : https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
Then you can execute cells **in the order of the notebook**
