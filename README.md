# Airbnb Classification Engine & Feature Optimization Pipeline

An end-to-end machine learning pipeline built using **Scikit-Learn** to predict Airbnb Superhost status. This project analyzes key property and host characteristics—such as response rates, ratings, and pricing—to classify hosts, optimizes model hyperparameters, and implements feature selection to maximize efficiency.

---

## 🚀 Project Overview
The goal of this project was to construct a robust classification model capable of predicting whether an Airbnb host qualifies for "Superhost" status. 

Through iterative development, the pipeline was enhanced from a baseline classifier to a tuned, dimensionally optimized model using hyperparameter grid search and statistical feature selection.

---

## 🛠️ Tech Stack & Key Libraries
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Model Persistence:** Pickle

---

## 📈 Pipeline Stages & Methodology

### 1. Data Preprocessing & Splitting
* Split the dataset into features ($X$) and target labels ($y$, representing Superhost status).
* Partitioned the data into training (70%) and testing (30%) sets to ensure rigorous evaluation.

### 2. Baseline Model
* Implemented a default `LogisticRegression` classifier to establish a baseline performance benchmark.
* Evaluated predictions using a confusion matrix to identify true positives, false positives, and misclassifications.

### 3. Hyperparameter Tuning
* Executed a systematic grid search (`GridSearchCV`) with 5-fold cross-validation.
* Evaluated 10 distinct values of the regularization strength parameter ($C$) to find the mathematically optimal setting to prevent overfitting.

### 4. Dimensionality Reduction & Feature Selection
* Utilized `SelectKBest` with the ANOVA F-value (`f_classif`) scoring function to rank feature importance.
* Experimented with varying feature counts ($k$) to analyze the impact of dimensionality on model performance:
  * **$k = 3$ features:** AUC = `0.7599` (Underfit; stripped away too much vital signal)
  * **$k = 5$ features:** AUC = `0.7926` (Moderate performance)
  * **$k = 10$ features:** AUC = **`0.8043`** (Optimal peak performance)

### 5. Serialization (Model Persistence)
* Exported the final, optimized model to a serialized byte stream (`model.pkl`) using Python's `pickle` library, allowing for instant downstream deployment without needing retraining.

---

## 📊 Final Performance Results
* **Optimal Hyperparameter:** `C = [Insert your best_C value here]`
* **Optimal Feature Count:** $k = 10$
* **Peak Model Accuracy:** **`0.8043` AUC-ROC**

The final ROC curve demonstrated that selecting the top 10 core features provided the ideal balance—capturing key statistical signals without introducing unnecessary dataset noise.

---

## 💻 How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
