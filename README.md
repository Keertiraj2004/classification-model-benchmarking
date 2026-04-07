<div align="center">

# 🤖 Classification Model Benchmarking

### A comparative study of supervised ML classification algorithms across real-world datasets

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Algorithms](https://img.shields.io/badge/Algorithms-6-purple?style=flat-square)
![Datasets](https://img.shields.io/badge/Datasets-2-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)

</div>

---

## 📌 Project Overview

This project implements and compares **6 supervised machine learning classification algorithms** on two well-known datasets — **Titanic** and **Diabetes** — covering the complete ML workflow from preprocessing to evaluation.

| | Detail |
|---|---|
| 📂 Datasets | Titanic (survival prediction) · Diabetes (diagnosis prediction) |
| 🛠️ Algorithms | 6 classifiers benchmarked |
| 📊 Metrics | Accuracy, Confusion Matrix, F1-Score |
| ✂️ Split | 80% train / 20% test |

---

## 🎯 Objectives

- Apply multiple classification algorithms on real-world datasets
- Preprocess and prepare data for model training
- Compare model performance using standard evaluation metrics
- Identify the best-performing model for each dataset

---

## 📂 Datasets

### 🚢 Titanic Dataset
Binary classification: *did the passenger survive?*

**Features used:** Age · Sex · Fare · Passenger Class · Embarked Port · Family Information

**Preprocessing applied:**
- Filled missing values in `Age`, `Embarked`, `Fare`
- Dropped: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Encoded: `Sex`, `Embarked`

---

### 💉 Diabetes Dataset
Binary classification: *does the patient have diabetes?*

**Features used:** Glucose · Blood Pressure · Insulin · BMI · Age · Pregnancies

---

## 🛠️ Algorithms Benchmarked

| # | Algorithm | Scaling Required |
|---|---|---|
| 1 | Logistic Regression | ✅ Yes |
| 2 | Support Vector Machine (SVM) | ✅ Yes |
| 3 | Random Forest Classifier | ❌ No |
| 4 | Naive Bayes | ✅ Yes |
| 5 | K-Nearest Neighbors (KNN) | ✅ Yes |
| 6 | Decision Tree Classifier | ❌ No |

> `StandardScaler` applied for algorithms that require normalized input.

---

## ⚙️ Workflow

```
1. Data Loading        → Load datasets via Pandas
2. Preprocessing       → Impute, drop, encode features
3. Feature Scaling     → StandardScaler for sensitive algorithms
4. Train/Test Split    → 80/20 split
5. Model Training      → Train all 6 classifiers
6. Evaluation          → Accuracy, Confusion Matrix, Classification Report
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy Score** | Percentage of correct predictions |
| **Confusion Matrix** | TP, TN, FP, FN breakdown per class |
| **Classification Report** | Precision, Recall, F1-Score per class |

---

## 📁 Project Structure

```bash
classification-model-benchmarking/
│
├── datasets/
│   ├── diabetes.csv
│   └── train.csv
│
├── notebooks/
│   └── classification_assignment.ipynb
│
├── python_files/
│   └── python_files.py
│
├── screenshots/
│   ├── Diabetes_Accuracy_Table.png
│   └── Titanic_Accuracy_Table.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Getting Started

### Step 1 — Clone the repository
```bash
git clone https://github.com/Keertiraj2004/classification-model-benchmarking.git
cd classification-model-benchmarking
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the notebook
```bash
jupyter notebook notebooks/classification_assignment.ipynb
```

Or run directly as a Python script:
```bash
python python_files/python_files.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
jupyterlab
notebook
```

---

## 📈 Results

Performance of all 6 classifiers was compared on both datasets.

| Dataset | Screenshot |
|---|---|
| Titanic | `screenshots/Titanic_Accuracy_Table.png` |
| Diabetes | `screenshots/Diabetes_Accuracy_Table.png` |

---

## 🚀 Planned Improvements

- [ ] Data visualization using Matplotlib and Seaborn
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for better reliability
- [ ] Feature importance analysis
- [ ] Model deployment with Streamlit or Flask
- [ ] Broader multi-dataset benchmarking

---

## 💡 Key Learnings

- How classification algorithms work under the hood
- The critical role of preprocessing in ML pipelines
- How model performance varies across different domains
- How to evaluate models using multiple metrics
- How to structure an ML project for GitHub and portfolio use

---

## 👨‍💻 Author

**Keertiraj Kamble**
Engineering Student · Aspiring Data Scientist · ML Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Keertiraj2004-black?style=flat-square&logo=github)](https://github.com/Keertiraj2004)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-keertiraj--kamble-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/keertiraj-kamble)

---

<div align="center">
  <sub>Developed as part of a Machine Learning Classification Assignment · MIT License</sub>
</div>
