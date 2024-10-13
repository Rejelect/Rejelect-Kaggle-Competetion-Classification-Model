---

# Kaggle Exam: Classification Model

This project contains a Jupyter Notebook that demonstrates building and evaluating a classification model using machine learning. The task appears to focus on predicting the likelihood of a customer exiting (or another binary outcome) based on a dataset of features. The model is evaluated using metrics like ROC AUC score.

## Project Structure

- **Data Loading**: The dataset is loaded using `pandas` and preprocessed by selecting relevant numeric columns for analysis.
  
- **Exploratory Data Analysis (EDA)**: 
  - **Correlation Heatmap**: A heatmap is generated to show the correlation between the numeric features and the target variable (`Exited`).
  
- **Modeling**: 
  - Various machine learning algorithms are tested, including:
    - Logistic Regression with L1 and L2 regularization
    - Ridge Classifier
  - Hyperparameter tuning is performed using `GridSearchCV` to find the best regularization parameters for these models.
  
- **Evaluation**: 
  - The performance of the models is evaluated using the **ROC AUC score**, which is printed during the grid search process.
  
- **Prediction and Submission**: 
  - The best model is used to make predictions on the test dataset. The predicted probabilities are then saved into a CSV file (`sub.csv`) for submission to Kaggle.

## Notable Code Sections

1. **Heatmap Generation**:
   ```python
   sns.heatmap(df.select_dtypes(include=['number']).corr(method="pearson")[["Exited"]], annot=True, cmap="cividis")
   ```
   - Generates a heatmap to visualize the correlation of numeric features with the target variable `Exited`.

2. **Modeling and Hyperparameter Tuning**:
   The project uses a **StackingClassifier** for combining different classifiers and performs a grid search for optimal hyperparameters:
   ```python
   estimators = [
       ('lr_l1', LogisticRegression(penalty='l1', solver='liblinear')),
       ('lr_l2', LogisticRegression(penalty='l2')),
       ('ridge', RidgeClassifier())
   ]

   param_grid = {
       'lr_l1__C': [9.0, 10.0, 11.0],
       'lr_l2__C': [0.1, 1, 2],
       'ridge__alpha': [0.1, 1, 2]
   }

   grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, scoring='roc_auc')
   grid_search.fit(X_train, y_train)
   ```

3. **Prediction and CSV Export**:
   The predicted probabilities for the test dataset are saved to a CSV file for Kaggle submission:
   ```python
   sub = pd.read_csv("submission.csv", index_col=0)
   sub["Exited"] = pipeline.predict_proba(test.select_dtypes(include=['number']).drop(['Tenure'], axis=1))[:,1]
   sub.to_csv("sub.csv")
   ```

## Requirements

- Python 3.12
- Jupyter Notebook
- Pandas
- Scikit-learn
- Seaborn

Install the necessary dependencies using:
```bash
pip install pandas scikit-learn seaborn
```

## Running the Notebook

1. Clone the repository or download the notebook file.
2. Install the necessary Python libraries.
3. Open the notebook using Jupyter:
   ```bash
   jupyter notebook exam_kaggle.ipynb
   ```


---

