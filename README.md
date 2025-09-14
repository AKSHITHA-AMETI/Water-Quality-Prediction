# Water Quality Prediction Using Machine Learning

## 📄 Project Overview
This project focuses on predicting the quality of water using machine learning techniques. Clean and safe drinking water is essential for human health, yet contamination is a global issue. By analyzing key water quality parameters, this project predicts whether a given water sample is safe for consumption.

This project was developed as part of a virtual internship in Green AI at Edunet.

---

## 🎯 Objectives
- To analyze water quality parameters and identify patterns in contaminated vs. clean water.
- To build and evaluate a machine learning model that classifies water as safe or unsafe.
- To provide an AI-driven solution for efficient water resource management and contamination detection.

---

## 🗂 Dataset
We used the **[Water Quality Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability/data)**.

- **Samples:** 3,276 water samples
- **Attributes:**
  - pH
  - Hardness
  - Solids
  - Chloramines
  - Sulfate
  - Conductivity
  - Organic Carbon
  - Trihalomethanes
  - Turbidity
  - Potability (Target: 1 = Safe, 0 = Unsafe)

---

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries:** 
  - pandas, numpy (data manipulation)
  - matplotlib, seaborn (visualization)
  - scikit-learn (machine learning algorithms)
- **Environment:** Jupyter Notebook / Google Colab

---

## 📈 Methodology
1. **Data Collection:** Downloaded dataset from kaggle .
2. **Data Preprocessing:**
   - Handle missing values
   - Normalize numerical features (if needed)
   - Split data into training and testing sets
3. **Model Selection:**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting (comparison)
4. **Model Training & Evaluation:**
   - Use accuracy, precision, recall, F1-score, and ROC-AUC metrics
   - Choose the best performing model
5. **Prediction:**
   - Predict water safety based on input parameters.

---
