# 🏦 Credit Risk Prediction using Machine Learning

## 📌 Problem Statement
Financial institutions face challenges in assessing the **creditworthiness** of loan applicants. This project uses the [German Credit Dataset](https://www.kaggle.com/datasets/uciml/german-credit) to build a machine learning model that classifies applicants as **good credit risk** or **bad credit risk**.

## 🎯 Objective
- Predict credit risk using various machine learning models.
- Select the best performing model.
- Build a UI using **Streamlit** for real-time prediction.
- Interpret model results and offer insights into feature importance.

---

## 🧠 Machine Learning Models Used

1. **XGBoost Classifier**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**

### ✅ Model Selection Criteria
Models are evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Classification report

The best-performing model is saved as `model.pkl`.

---

## ⚙️ Project Structure

### Model Development:
- Select appropriate machine learning algorithms for classification.
- Train and validate the model using suitable evaluation metrics (e.g., accuracy, precision, recall, F1-score).
- Optimize the model through techniques such as hyperparameter tuning and cross-validation.

### Model Interpretation and Insights:
- Interpret the model's predictions and identify the most influential features.
- Create visualizations to communicate findings effectively.
- Provide actionable insights and recommendations for improving the credit evaluation process.

Additionally, a **Streamlit** UI is provided for easy deployment and prediction.

## Dataset Description

The dataset contains the following features:

- **Age** (numeric)
- **Sex** (text: male, female)
- **Job** (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
- **Housing** (text: own, rent, free)
- **Saving accounts** (text - little, moderate, quite rich, rich)
- **Checking account** (numeric, in DM - Deutsch Mark)
- **Credit amount** (numeric, in DM)
- **Duration** (numeric, in months)
- **Purpose** (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

The target variable is **Risk**, with two possible values:
- **Good** (1)
- **Bad** (0)

---

## Approach

### Step 1: Data Preprocessing
The following steps were taken:
- **Outlier Handling**: Outliers in numeric features were capped using the Interquartile Range (IQR) method.
- **Encoding**: Categorical variables were mapped to numerical values:
  - Sex: 'male' = 0, 'female' = 1
  - Housing: 'own' = 0, 'rent' = 1, 'free' = 2
  - Saving accounts: 'little' = 0, 'moderate' = 1, 'quite rich' = 2, 'rich' = 3
  - Checking account: 'little' = 0, 'moderate' = 1, 'rich' = 2
  - Purpose: Categorical values were encoded using LabelEncoder.
  
### Step 2: Feature Selection
- **Mutual Information** was used to select the most important features.
- **Recursive Feature Elimination (RFE)** with Logistic Regression was used to further reduce features, selecting the top 8 features based on importance.

### Step 3: Model Training
Three machine learning models were evaluated:
1. **SVM (Support Vector Machine)**
2. **Random Forest Classifier**
3. **K-Nearest Neighbors (KNN)**

The model performance was evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Step 4: Model Evaluation
The model with the best evaluation metric score was selected as the final model.

### Step 5: Saving Models and Resources
The best-performing model, scaler, and selected features were saved as `.pkl` files for future use:
- `model.pkl` (Trained model)
- `scaler.pkl` (Scaler used for feature scaling)
- `selected_features.pkl` (List of selected features)

---

## Code Files

### 1. **ML Training Script (`ml_training.py`)**
The `ml_training.py` script performs the following:
- Loads and preprocesses the dataset.
- Handles missing values and encodes categorical features.
- Performs feature selection using Mutual Information and RFE.
- Trains and evaluates models (SVM, Random Forest, KNN).
- Saves the best model, scaler, and selected features as `.pkl` files.

### 2. **Streamlit App (`streamlit_app.py`)**
The `streamlit_app.py` provides a user interface where users can input loan applicant data and receive a credit risk prediction. It:
- Loads the saved model and scaler.
- Accepts user input (age, credit amount, purpose, etc.).
- Applies encoding to categorical inputs and scales the features.
- Makes a prediction (Good or Bad Credit Risk).
- Displays the result in the Streamlit interface.

### 3. **Model and Resources**
- **`model/model.pkl`**: The trained machine learning model.
- **`model/scaler.pkl`**: The scaler used for feature normalization.
- **`model/selected_features.pkl`**: The list of selected features used for prediction.

---


