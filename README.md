# Advanced Categorical Feature Encoding with CatBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-CatBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project addresses the challenge of binary classification on high-cardinality categorical data using the **Kaggle "Cat in the Dat II"** dataset. The primary objective is to accurately predict target probabilities while handling missing values, complex ordinal features, and potential data leakage.

The solution utilizes a Hybrid Feature Engineering strategy combined with a GPU-accelerated **CatBoost Classifier**, achieving a robust AUC score through Stratified K-Fold Cross-Validation.

## 📊 Data & Objective
* **Dataset:** Categorical Feature Encoding Challenge II (Kaggle).
* **Problem Type:** Binary Classification.
* **Challenge:** The dataset consists entirely of categorical variables (nominal, ordinal, binary, and cyclical) with missing values and high cardinality.

## 🛠️ Methodology & Feature Engineering
This project relies heavily on advanced preprocessing to maximize model performance:

1.  **Missing Value Imputation:** * Created binary flags (`_was_missing`) to track data absence patterns.
    * Imputed numeric gaps with `-1` and categorical gaps with `'NONE'` to preserve information.
2.  **Ordinal Encoding:** Mapped qualitative scales (e.g., "Novice", "Grandmaster") to numerical hierarchies.
3.  **Cyclical Transformation:** Transformed temporal features (`day`, `month`) using Sine/Cosine features to preserve cyclical continuity.
4.  **Feature Interactions:** Created new features by combining high-impact variables (e.g., `ord_1` + `nom_3`) to capture non-linear relationships.
5.  **Frequency Encoding:** Mapped categorical frequencies to capture the "popularity" of specific categories.

## 🧠 Model Architecture
* **Algorithm:** CatBoostClassifier (Gradient Boosting on Decision Trees).
* **Hardware Acceleration:** GPU (`task_type='GPU'`).
* **Hyperparameters:**
    * `iterations`: 3000
    * `learning_rate`: 0.02
    * `depth`: 6
* **Validation Strategy:** Stratified K-Fold (10 Splits) to maintain class balance across folds.

## 📈 Results
The model was evaluated using the Area Under the Curve (AUC) metric on Kaggle.

| Metric | Score |
| :--- | :--- |
| **Public Score** | **0.78555** |
| **Private Score** | **0.78688** |

*Note: The consistency between Public and Private scores demonstrates the model's stability and lack of overfitting.*


## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Advanced-Categorical-Encoding-Classifier.git](https://github.com/yourusername/Advanced-Categorical-Encoding-Classifier.git)