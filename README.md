# Facial Expression Detection using CNN

This repository implements a lightweight Convolutional Neural Network (CNN) in PyTorch to classify facial expressions using grayscale images from the FER2013 and RAF-DB datasets. The training pipeline includes real-time progress monitoring, confusion matrix visualization, and classification performance export for further evaluation.

---

## 📚 Datasets

This project uses two publicly available datasets:

- **FER2013**  
  - [Kaggle: FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
  - Contains grayscale images of size 48x48 with 7 emotion classes.

- **RAF-DB (Real-world Affective Faces Database)**  
  - [RAF-DB Official Site](http://www.whdeng.cn/RAF/model1.html)
  - High-quality facial expression dataset with annotated basic and compound emotions.

---

## 🧠 Model Architecture

The CNN model architecture is intentionally simple for efficiency and fast convergence:

