# 👤 Face Recognition: PCA vs LDA (Eigenfaces & Fisherfaces)

A comparative study of dimensionality reduction and classification techniques for facial recognition using the **ORL Face Dataset**.

## 📌 Overview
This project implements and evaluates two classical subspace projection methods combined with a K-Nearest Neighbors (KNN) classifier:
* **PCA (Eigenfaces):** Unsupervised dimensionality reduction maximizing global variance across all face images.
* **LDA (Fisherfaces):** Supervised dimensionality reduction maximizing between-class scatter while minimizing within-class scatter.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `NumPy`, `Scikit-learn`, `OpenCV`, `Matplotlib`

## 📊 Key Results
* Comparison of recognition accuracy across varying numbers of principal/discriminant components.
* Visualization of extracted Eigenfaces, Fisherfaces, and confusion matrices.

## 🚀 How to Run
1. Clone the repository:
   ```bash
git clone https://github.com/kniza23/face-recognition-pca-lda.git
