# 🖊️ MNIST Handwritten Digit Recognition

This project implements and compares multiple **machine learning classifiers** on the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
The goal is to classify handwritten digits (0–9) from 28×28 grayscale images.

---

## 📊 Dataset
- **Source**: [MNIST 784 from OpenML](https://www.openml.org/d/554)
- **Size**: 70,000 images (60,000 train + 10,000 test)
- **Features**: 784 pixels (28×28 flattened grayscale images)
- **Labels**: 10 classes (digits 0–9)

---

## ⚙️ Project Workflow
1. **Data Loading & Exploration**  
   - Load MNIST dataset via `fetch_openml`
   - Check for missing values
   - Visualize class distribution and sample digits

2. **Data Preprocessing**  
   - Feature scaling: normalize pixel values to [0–1] range  
   - Train-test split: 80% training, 20% testing

3. **Model Building & Evaluation**  
   - Implement multiple classifiers
   - Train on training set and evaluate on test set
   - Use accuracy, confusion matrix, and classification report

---

## 🧑‍💻 Implemented Models & Results

| Model                     | Training Accuracy | Test Accuracy | Notes |
|----------------------------|------------------|---------------|-------|
| **Support Vector Machine (Linear)** | 0.9730 | 0.9353 | Good balance, slight variance. |
| **K-Nearest Neighbors (KNN)**       | 0.9846 | 0.9674 | Strong performance, but costly on large datasets. |
| **Logistic Regression**             | 0.9361 | 0.9213 | Stable, interpretable, slightly weaker accuracy. |
| **Random Forest**                   | 1.0000 | 0.9677 | Best performance, but risk of overfitting (perfect training score). |
| **Decision Tree**                   | 0.9833 | 0.8743 | Overfitting observed (large gap between train & test). |
| **Naïve Bayes (BernoulliNB)**       | 0.8332 | 0.8344 | Weakest performer due to independence assumption. |

---

## 🔍 Key Observations
- **Best Test Accuracy**: Random Forest (0.9677) and KNN (0.9674)  
- **Most Balanced Model**: SVM (Linear) with good generalization  
- **Overfitting Models**: Decision Tree and Random Forest (train ≫ test)  
- **Simplest & Interpretable**: Logistic Regression  
- **Poor Fit**: Naïve Bayes due to strong independence assumptions  

---

## 📉 Visualizations
- Class distribution of digits
- Sample digit images
- Confusion matrices for each classifier
- Heatmaps for model evaluation

---

## 🚀 Future Work
- Apply **deep learning (CNNs)** for higher accuracy  
- Perform **hyperparameter tuning** (GridSearchCV / RandomizedSearchCV)  
- Use **dimensionality reduction (PCA, t-SNE)** for faster computation  
- Try **ensemble methods** to combine strengths of classifiers  

---

## 🛠️ Tech Stack
- **Python 3**
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- **Environment**: Google Colab / Jupyter Notebook

---

## 📌 How to Run
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/mnist-handwritten-digits.git
   cd mnist-handwritten-digits
