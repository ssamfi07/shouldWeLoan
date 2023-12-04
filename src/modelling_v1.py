import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  confusion_matrix,accuracy_score, classification_report
from sklearn.utils import resample


import utils

# V1
# ----------------------------------------------------------------
# Predictive modelling based on the features identified as relevant - logistic regression & naive bayes applied
# ----------------------------------------------------------------

def train_logistic_regression(loan_features, features):

    # Target variable: 'status'
    target = 'status'

    # Drop rows with NaN values (if any)
    df_cleaned = loan_features.dropna(subset=features + [target])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_rep)

    return model, scaler

def predict_model(model, scaler, loan_features, features):

    # Drop rows with NaN values (if any)
    df_cleaned = loan_features.dropna(subset=features)

    # Standardize the features
    X_scaled = scaler.transform(df_cleaned[features])

    # Predict probabilities on the new dataset
    probabilities = model.predict_proba(X_scaled)[:, 0]  # Probability of status being -1

    # Prepare a DataFrame with loan_id and predicted probabilities
    result_df = pd.DataFrame({'Id': df_cleaned['loan_id'], 'Predicted': probabilities})

    return result_df

def train_naive_bayes(loan_features, features):
    # Target variable: 'status'
    target = 'status'

    # Drop rows with NaN values (if any)
    df_cleaned = loan_features.dropna(subset=features + [target])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    # Predict probabilities on the test set
    probabilities = model.predict_proba(X_test_scaled)[:, 0]  # Probability of status being -1

    # Evaluate the model
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_rep)

    return model, scaler

def apply_model_and_export(df_trans_train, df_loans_train, df_trans_predict, df_loans_predict):
    # relevant features
    features = ['amount_std', 'amount_mean', 'num_transactions', 'total_withdrawn', 'balance_std', 'balance_mean', 'amount_loan', 'duration']

    # dataframes
    df_trans_sorted, df_loans_sorted = utils.sort_trans_loans_by_account_id(df_trans_train, df_loans_train)
    kaggle_df_trans_sorted, kaggle_df_loans_sorted = utils.sort_trans_loans_by_account_id(df_trans_predict, df_loans_predict)

    kaggle_features = utils.aggregation(kaggle_df_trans_sorted, kaggle_df_loans_sorted)
    loan_features = utils.aggregation(df_trans_sorted, df_loans_sorted)

    print(kaggle_features)

    # number of loans
    num_loans = loan_features['status'].value_counts()
    print(f'Number of accounts based on status: {num_loans}')

    # separate majority (status 1) and minority (status -1) classes
    majority_class = loan_features[loan_features['status'] == 1]
    minority_class = loan_features[loan_features['status'] == -1]

    # undersample majority class
    majority_undersampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)

    # combine minority class with undersampled majority class
    undersampled_data = pd.concat([majority_undersampled, minority_class])

    # training Phase with the data from moodle
    logistic_regression_model, logistic_regression_scaler = train_logistic_regression(loan_features, features)
    naive_bayes_model, naive_bayes_scaler = train_naive_bayes(loan_features, features)
    # prediction Phase
    logistic_regression_predictions = predict_model(logistic_regression_model, logistic_regression_scaler, kaggle_features, features)
    naive_bayes_predictions = predict_model(naive_bayes_model, naive_bayes_scaler, kaggle_features, features)
    # print(logistic_regression_predictions)
    # print(naive_bayes_result)
    # print(logistic_regression(loan_features))
    # print(naive_bayes_classifier(loan_features))
    # export to csv
    utils.simple_export(logistic_regression_predictions, "linear_regression_result.csv")
    utils.simple_export(naive_bayes_predictions, "naive_bayes_result.csv")