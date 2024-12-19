# Fraud Detection Project

This project focuses on detecting fraudulent transactions using machine learning techniques. It includes data preprocessing, feature engineering, handling class imbalance, and applying various algorithms to achieve the best performance.

---

## **Project Workflow**

### 1. **Data Loading**
- Load the transaction data from a compressed CSV file.
- Example:
  ```python
  df1 = pd.read_csv("/content/transactions.gz")
  ```

### 2. **Exploratory Data Analysis (EDA)**
- Inspect the dataset using methods like `head()`, `info()`, and `value_counts()`.
- Key checks:
  - Data types and null values.
  - Distribution of the target variable (`isFraud`).

### 3. **Data Cleaning**
- Handle missing values:
  - Categorical columns: Replace missing values with 'unknown'.
  - Numerical columns: Fill with the column mean.
- Drop irrelevant columns such as `echoBuffer`, `merchantCity`, `merchantState`, etc.

### 4. **Feature Engineering**
- Extract temporal features from the `transactionDateTime` column:
  - Year, Month, Day, Hour, Minute, Second.
- Create new features:
  - `UtilizationRate`: (current balance + transaction amount) / credit limit.
  - `RemainingCredit`: credit limit - (current balance + transaction amount).
  - `TransactionToBalanceRatio`: transaction amount / (current balance).
  - `TransactionFrequency`: difference between transaction times.

### 5. **Feature Selection and Encoding**
- Drop columns with low relevance (e.g., `accountNumber`, `transactionDateTime`, etc.).
- Use one-hot encoding for categorical variables.
  ```python
  dummy = pd.get_dummies(df1, columns=['acqCountry', 'merchantCountryCode', 'merchantCategoryCode', 'transactionType'], drop_first=True)
  ```

### 6. **Handling Class Imbalance**
- Apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance the `isFraud` classes in the training data.
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42)
  x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
  ```

### 7. **Modeling**
#### a. Logistic Regression
  model = LogisticRegression()
  model.fit(x_train_res_scaled, y_train_res)
  ```

#### b. Random Forest
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(class_weight='balanced')
  model.fit(x_train_res, y_train_res)
  ```

#### c. XGBoost
  import xgboost as xgb
  params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'scale_pos_weight': len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])}
  model_xgb = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "Test")], early_stopping_rounds=10)
  ```

#### d. LightGBM
  import lightgbm as lgb
  params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05}
  model_lgb = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_test])
  ```

### 8. **Evaluation Metrics**
- **Accuracy**: Overall correctness of predictions.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **AUC-ROC**: Model's ability to distinguish between classes.
  ```python
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred))
  print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
  ```

---

## **Technologies Used**
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, Seaborn.
- **Machine Learning Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM.

---

## **Project Highlights**
1. Extensive preprocessing and feature engineering.
2. Addressed severe class imbalance using SMOTE and algorithm-specific techniques.
3. Evaluated multiple algorithms to ensure robust fraud detection.
4. Achieved high accuracy and recall for the minority (fraudulent) class.

---

## **Future Work**
- Implement additional techniques for explainability (e.g., SHAP).
- Explore real-time fraud detection capabilities.
- Fine-tune hyperparameters for further performance gains.

