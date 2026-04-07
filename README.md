# Classification Model Benchmarking using Machine Learning

## 📌 Project Overview
This project focuses on the **implementation and comparison of multiple supervised machine learning classification algorithms** on two popular Kaggle datasets:

- **Titanic Dataset**
- **Diabetes Dataset**

The main objective of this project is to analyze how different classification algorithms perform on different datasets and evaluate them using standard machine learning metrics.

This project demonstrates the complete machine learning workflow including:

- Data loading
- Data preprocessing
- Feature transformation
- Model training
- Performance evaluation
- Comparative analysis of classification algorithms

---

## 🎯 Objectives
- To apply machine learning classification algorithms on real-world datasets
- To preprocess and prepare datasets for model training
- To compare the performance of multiple classification models
- To evaluate models using standard performance metrics
- To identify the best-performing model for each dataset

---

## 📂 Datasets Used

### 1️⃣ Titanic Dataset
The Titanic dataset is used to predict whether a passenger survived or not based on features such as:

- Age
- Sex
- Fare
- Passenger Class
- Embarked Port
- Family Information

### 2️⃣ Diabetes Dataset
The Diabetes dataset is used to predict whether a patient is likely to have diabetes based on medical attributes such as:

- Glucose
- Blood Pressure
- Insulin
- BMI
- Age
- Pregnancies

---

## 🛠️ Machine Learning Algorithms Used
The following classification algorithms were implemented and compared:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

---

## ⚙️ Technologies & Libraries Used

- **Python**
- **Jupyter Notebook / JupyterLab**
- **Pandas**
- **NumPy**
- **Scikit-learn**

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
🔍 Project Workflow
1️⃣ Data Loading

Both datasets are loaded using Pandas.

2️⃣ Data Preprocessing
Titanic Dataset Preprocessing:
Filled missing values in:
Age
Embarked
Fare
Dropped unnecessary columns:
PassengerId
Name
Ticket
Cabin
Encoded categorical columns:
Sex
Embarked
Diabetes Dataset Preprocessing:
Loaded and prepared the dataset for classification
Selected features and target variable
3️⃣ Feature Scaling

Applied StandardScaler for algorithms that require normalized input:

Logistic Regression
SVM
Naive Bayes
KNN
4️⃣ Train-Test Split

Both datasets were split into:

80% Training Data
20% Testing Data
5️⃣ Model Training

Each classification model was trained separately on both datasets.

6️⃣ Model Evaluation

The performance of each model was evaluated using:

Accuracy Score
Confusion Matrix
Classification Report
📊 Evaluation Metrics
✅ Accuracy Score

Measures the percentage of correct predictions made by the model.

✅ Confusion Matrix

Shows the number of:

True Positives
True Negatives
False Positives
False Negatives
✅ Classification Report

Provides detailed metrics such as:

Precision
Recall
F1-Score
📈 Results

The performance of all classification algorithms was compared on both datasets.

Titanic Dataset

A comparative accuracy table was generated for:

Logistic Regression
SVM
Random Forest
Naive Bayes
KNN
Decision Tree
Diabetes Dataset

A similar comparative accuracy analysis was performed for the Diabetes dataset.

📸 Project Screenshots
Titanic Dataset Accuracy Table

Diabetes Dataset Accuracy Table

▶️ How to Run This Project
Step 1: Clone the Repository
git clone https://github.com/your-username/classification-model-benchmarking.git
cd classification-model-benchmarking
Step 2: Install Required Libraries
pip install -r requirements.txt
Step 3: Run the Jupyter Notebook
jupyter notebook

Then open:

notebooks/classification_assignment.ipynb

OR run the Python script directly:

python python_files/python_files.py
📦 requirements.txt

The following libraries are required to run this project:

pandas
numpy
scikit-learn
jupyterlab
notebook
💡 Key Learnings

This project helped in understanding:

How classification algorithms work
The importance of preprocessing in machine learning
How model performance varies across datasets
How to evaluate classification models using different metrics
How to structure a machine learning project for GitHub and portfolio use
🚀 Future Improvements

This project can be improved further by adding:

Data visualization using Matplotlib and Seaborn
Hyperparameter tuning using GridSearchCV
Cross-validation for better reliability
Feature importance analysis
Model deployment using Streamlit or Flask
More datasets for broader benchmarking
🎓 Academic Relevance

This project was developed as part of a Machine Learning Classification Assignment to gain practical hands-on experience with supervised learning techniques.

👨‍💻 Author

Keertiraj
Engineering Student | Aspiring Data Scientist | Machine Learning Enthusiast

Connect with me:
GitHub: https://github.com/Keertiraj2004
LinkedIn: www.linkedin.com/in/keertiraj-kamble
