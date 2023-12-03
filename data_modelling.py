import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

df_account = pd.read_csv('bank/account.csv', sep=';', low_memory=False)
df_client = pd.read_csv('bank/client.csv', sep=';', low_memory=False)
df_disp = pd.read_csv('bank/disp.csv', sep=';', low_memory=False)
df_trans = pd.read_csv('bank/trans_dev.csv', sep=';', low_memory=False)
df_loan = pd.read_csv('bank/loan_dev.csv', sep=';', low_memory=False)
df_card = pd.read_csv('bank/card_dev.csv', sep=';', low_memory=False)
df_district = pd.read_csv('bank/district.csv', sep=';', low_memory=False)

df_trans_comp = pd.read_csv('bank/trans_comp.csv', sep=';', low_memory=False)
df_loans_comp = pd.read_csv('bank/loan_comp.csv', sep=';', low_memory=False)

# ----------------------------------------------------------------
# sort loans and transactions by date and ids
# ----------------------------------------------------------------

# sort by account ids
df_accounts_sort = df_account.sort_values('account_id')
# print(df_accounts_sort)

def sort_trans_loans_by_account_id(df_trans, df_loan):
    # sort by account and date
    df_trans_sorted = df_trans.sort_values(by=["account_id", "date"])
    # print(df_trans_sorted[df_trans_sorted['account_id'] == 19])

    # Get unique account_id values
    unique_account_ids = df_trans_sorted['account_id'].unique()

    # sort by account and date
    df_loans_sorted = df_loan.sort_values(by=["account_id", "date"])

    # Format the date to epoch time so it's easier to work with later
    df_trans_sorted['date'] = pd.to_datetime(df_trans_sorted['date'], format='%y%m%d').astype(int) // 10**9

    # Replace "withdrawal in cash" with "withdrawal" for all types
    df_trans_sorted['type'] = df_trans_sorted['type'].replace('withdrawal in cash', 'withdrawal')
    # print(df_trans_sorted)

    # Calculate the date difference for each transaction type within each account
    df_trans_sorted['date_diff'] = df_trans_sorted.groupby(['account_id', 'type'])['date'].diff()

    return df_trans_sorted, df_loans_sorted

# for the original database.csv
def export_trans_loans_merge(df_trans, df_loan, file_name):
    df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loan)
    merged = pd.merge(df_loans_sorted, df_trans_sorted, on='account_id', how='left')
    merged.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id'], axis=1, inplace=True)
    merged.rename(columns={'amount_x': 'amount_loan', 'amount_y': 'amount_trans'}, inplace=True)
    merged.to_csv(file_name, sep=',', index=False, encoding='utf-8')

def plot_balance_graphs(unique_account_ids, df_trans_sorted, df_loans_sorted):
    # for all accounts make the balance graph in time based in transactions - deposits(lines) or expenditures(dots)
    for account in unique_account_ids:
        # print(account)
        df_account = df_trans_sorted[df_trans_sorted['account_id'] == account]

        # Plot 2: Amount with different colors for each type of operation - deposit or expenditure
        plt.figure(figsize=(12, 8))
        colors = {'credit in cash': 'green', 'withdrawal in cash': 'red',
                'collection from another bank': 'blue', 'remittance from another bank': 'orange', 'NaN': 'gray'}

        plt.plot(df_account['date'], df_account['balance'], label='current_balance', color='pink')
        # print("Operations for account" + str(account))
        for operation, color in colors.items():
            operation_data = df_account[df_account['operation'] == operation]
            # print(operation)
            if operation == 'collection from another bank':
                for date in operation_data['date']:
                    plt.axvline(date, color=color, linestyle='--', alpha=0.5)
                # plt.plot(operation_data['date'], operation_data['amount'], label=operation, color=color)
            elif operation == 'credit in cash':
                for date in operation_data['date']:
                    plt.axvline(date, color=color, linestyle='--', alpha=0.5)
            else:
                # mean_balance_per_date = operation_data.groupby('date')['balance'].mean()
                # mean_balance_per_date.plot(marker='o', linestyle='-', color=color)
                
                plt.scatter(operation_data['date'], operation_data['amount'], label=operation, color=color)

        if account in df_loans_sorted['account_id'].values:
            # Use boolean indexing to filter rows based on the condition
            subset_df = df_loans_sorted[df_loans_sorted['account_id'] == account]
            print(subset_df['payments'])
            # if (subset_df['account_id'] == ).all():
            #     plt.axhline(y=subset_df['payments'][158], color='red', linestyle='--', alpha=0.5, label='montly loan payment')
            if (subset_df['status'] == 1).all(): # because there is only one value anyway
                plt.title('Balance for Each Type of Operation for account_id == ' + str(account) + " LOAN PAID")
            elif (subset_df['status'] == -1).all():
                plt.title('Balance for Each Type of Operation for account_id == ' + str(account) + " LOAN UNPAID")
        else:
            plt.title('Amount with Different Colors for Each Type of Operation for account_id == ' + str(account) + "NO LOAN")

        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.legend()
        plt.show()

def feature_engineering(df_loans_sorted, df_trans_sorted):
    # merge transactions with loans based on account id
    df_loan_trans_account = pd.merge(df_loans_sorted, df_trans_sorted, on='account_id', how='inner')
    # we will drop some columns -- date_x corresponds to the issuing date of the loan, it is always after the transaction history ends
    df_loan_trans_account.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id'], axis=1, inplace=True)

    # handle missing values in date_diff
    df_loan_trans_account['date_diff'].fillna(0, inplace=True)

    # maybe for amount of transaction, if the type is withdrawal we need to set the amount to the opposite value (negative)
    df_loan_trans_account['amount_trans'] = df_loan_trans_account.apply(lambda row: -row['amount_y'] if row['type'] == 'withdrawal' else row['amount_y'], axis=1)

    # print(df_loan_trans_account)

    # Aggregating transaction-level data to the loan level
    loan_data = df_loan_trans_account.groupby(['loan_id', 'account_id'])[['amount_trans', 'balance']].agg({
        'amount_trans': ['std', 'mean', 'count', lambda x: np.abs(x[x < 0].sum())],
        'balance': ['std','mean']
    }).reset_index()

    loan_data.columns = ['loan_id', 'account_id', 'amount_std', 'amount_mean', 'num_transactions', 'total_withdrawn', 'balance_std', 'balance_mean']

    # rename column to correct description
    df_loan_trans_account.rename(columns={'amount_y': 'amount_loan'}, inplace=True)

    # Create a DataFrame with unique values for each loan_id
    unique_loan_info = df_loan_trans_account[['loan_id', 'amount_loan', 'duration', 'status']].drop_duplicates()

    # Merge loan_data with unique_loan_info based on loan_id
    loan_data = pd.merge(loan_data, unique_loan_info, on='loan_id', how='left')

    return loan_data

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

def apply_model_and_export():
    # relevant features
    features = ['amount_std', 'amount_mean', 'num_transactions', 'total_withdrawn', 'balance_std', 'balance_mean', 'amount_loan', 'duration']

    # dataframes
    df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loan)
    kaggle_df_trans_sorted, kaggle_df_loans_sorted = sort_trans_loans_by_account_id(df_trans_comp, df_loans_comp)

    kaggle_features = feature_engineering(kaggle_df_trans_sorted, kaggle_df_loans_sorted)
    loan_features = feature_engineering(df_trans_sorted, df_loans_sorted)

    print(kaggle_features)

    # number of loans
    num_loans = loan_features['status'].value_counts()
    print(f'Number of accounts based on status: {num_loans}')

    # Separate majority (status 1) and minority (status -1) classes
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
    logistic_regression_predictions.to_csv("linear_regression_result.csv", sep=',', index=False, encoding='utf-8')
    naive_bayes_predictions.to_csv("naive_bayes_result.csv", sep=',', index=False, encoding='utf-8')

# apply_model_and_export()

def merge_and_export(df_trans, df_loans, fileName):
    df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loans)
    total = feature_engineering(df_trans_sorted, df_loans_sorted)
    # drop account_id, we already have loan_id
    total.drop('account_id', axis=1, inplace=True)
    total.to_csv(fileName, sep=',', index=False, encoding='utf-8')
    print(total)

merge_and_export(df_trans, df_loan, "database.csv")