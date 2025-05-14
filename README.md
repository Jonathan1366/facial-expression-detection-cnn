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

Input: 48x48 grayscale image
- Conv2d(1, 32, kernel=3, padding=1) → ReLU → MaxPool2d(2)
- Conv2d(32, 64, kernel=3, padding=1) → ReLU → MaxPool2d(2)
- Flatten → Linear(641212, 128) → ReLU → Linear(128, num_classes)


---


## 🚀 Features

- ✅ Multi-dataset training (FER2013 and RAF-DB)
- ✅ Real-time training loop with `tqdm`
- ✅ Automatic CSV export of accuracy & loss per epoch
- ✅ Confusion matrix visualization (PNG)
- ✅ Classification report export (CSV)
- ✅ Trained model export (`.pt`) for future inference


---


## 🛠️ Setup and Training


### 1. Clone the Repository

git clone https://github.com/jonathan1366/facial-expression-detection-cnn.git
cd facial-expression-detection-cnn


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Dataset Structure

Ensure your dataset folders are structured like this:

<img width="179" alt="image" src="https://github.com/user-attachments/assets/ffa1387d-4f29-45c6-a940-a5ae0ac5f973" />

### 4. Run the Training Script

python train.py


---


## 📦 Output Files

After training, the following files will be generated:

| File name                           | Description                                |
| ----------------------------------- | ------------------------------------------ |
| `accuracy_plot_fer2013.png`         | Accuracy curve across epochs (FER2013)     |
| `loss_plot_fer2013.png`             | Loss curve across epochs (FER2013)         |
| `classification_report_fer2013.csv` | Precision, Recall, F1-score for each class |
| `confusion_matrix_fer2013.png`      | Heatmap of predicted vs actual labels      |
| `fer2013_model.pt`                  | Trained PyTorch model weights              |
| (and the same for RAF-DB)           |                                            |


📈 Sample Evaluation Metrics

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Angry | 0.71      | 0.68   | 0.69     |
| Happy | 0.88      | 0.90   | 0.89     |
| Sad   | 0.74      | 0.70   | 0.72     |
| ...   | ...       | ...    | ...      |


🔍 Use Case

This model can be used for:

- Facial expression-based emotion detection
- Behavioral analysis in HCI (Human-Computer Interaction)
- Real-time camera-based affective computing (requires additional optimization)



