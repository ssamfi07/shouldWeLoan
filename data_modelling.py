import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pingouin as pg

from category_encoders import BinaryEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix,accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier

import scipy
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr


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
    merged.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id', 'account_id'], axis=1, inplace=True)
    merged.rename(columns={'amount_x': 'amount_loan', 'amount_y': 'amount_trans'}, inplace=True)
    merged.to_csv(file_name, sep=',', index=False, encoding='utf-8')

def plot_balance_graphs(df_trans_sorted, df_loans_sorted):
    # for all accounts make the balance graph in time based in transactions - deposits(lines) or expenditures(dots)
    for account in range(1,10000):
        df_account = df_trans_sorted[df_trans_sorted['account_id'] == account]

        print(df_account)

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

def merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted):
    # rename columns
    df_loans_sorted.rename(columns={'amount': 'amount_loan'}, inplace=True)
    df_trans_sorted.rename(columns={'amount': 'amount_trans'}, inplace=True)
    # merge transactions with loans based on account id
    df_loan_trans_account = pd.merge(df_loans_sorted, df_trans_sorted, on='account_id', how='inner')
    # we will drop some columns -- date_x corresponds to the issuing date of the loan, it is always after the transaction history ends
    df_loan_trans_account.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id'], axis=1, inplace=True)

    # handle missing values in date_diff
    df_loan_trans_account['date_diff'].fillna(0, inplace=True)

    # maybe for amount of transaction, if the type is withdrawal we need to set the amount to the opposite value (negative)
    df_loan_trans_account['amount_trans'] = df_loan_trans_account.apply(
        lambda row: -row['amount_trans'] if row['type'] == 'withdrawal' else row['amount_trans'],
        axis=1)

    return df_loan_trans_account

def aggregation(df_loan_trans_account):
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

def merge_and_export(df_trans, df_loans, fileName):
    df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loans)
    total = aggregation(df_trans_sorted, df_loans_sorted)
    # drop account_id, we already have loan_id
    total.drop('account_id', axis=1, inplace=True)
    total.to_csv(fileName, sep=',', index=False, encoding='utf-8')
    print(total)

def simple_export(df, fileName):
    df.to_csv(fileName, sep=',', index=False, encoding='utf-8')

def accounts_with_disponents(df_loan_trans_account, df_disponents):
    unique_ids_loan = df_loan_trans_account['account_id'].unique().tolist()
    df_disp_filtered = df_disponents[df_disponents['account_id'].isin(unique_ids_loan)]
    accounts_with_disponent = df_disp_filtered[df_disp_filtered['type'] == 'DISPONENT']['account_id'].tolist()
    return accounts_with_disponent

# too little info from this...we only have 11 accounts with credit cards
# adds the column card type for the accounts that have a card
def accounts_with_disponents_and_card(df_loan_trans_account, df_disponents, df_card):
    # first sort the card entries based on disp_id
    df_card = df_card.sort_values(by=["disp_id"])
    df_card_disp = pd.merge(df_card, df_disponents, on='disp_id', how='inner')
    df_card_disp = df_card_disp.sort_values(by=["disp_id"])

    unique_ids_loan = df_loan_trans_account['account_id'].unique().tolist()
    df_disp_filtered = df_disponents[df_disponents['account_id'].isin(unique_ids_loan)]

    unique_disp_ids = df_disp_filtered['disp_id'].unique().tolist()
    # drop rows that don't have the disp_id in unique_disp_ids
    df_card_disp_filtered = df_card_disp[df_card_disp['disp_id'].isin(unique_disp_ids)]
    df_card_disp_filtered.drop(['card_id', 'disp_id', 'issued', 'client_id', 'type_y'], axis=1, inplace=True)
    df_card_disp_filtered.rename(columns={'type_x': 'card_type'}, inplace=True)
    print(df_card_disp_filtered)
    # accounts_with_disponent = df_disp_filtered[df_disp_filtered['type'] == 'DISPONENT']['account_id'].tolist()
    # return accounts_with_disponent

# 1 for account with disponent, 0 for account without disponent
def add_disponent_info_to_loan_trans(df_loan_trans_account, accounts_with_disponent):
    df_loan_trans_account['disp'] = df_loan_trans_account.apply(
        lambda row: "disponent" if row['account_id'] in accounts_with_disponent else "no_disponent",
        axis=1)
    return df_loan_trans_account

def transform_status(df_loan_trans_account):
    df_loan_trans_account['status'] = df_loan_trans_account.apply(
        lambda row: 1 if row['status'] == 1 else 0,
        axis=1)
    return df_loan_trans_account

def pie_chart_card_types(df_card):
    # Count the occurrences of each type
    type_counts = df_card['type'].value_counts()

    # Plot a pie chart
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Card Types')
    plt.show()

def pie_chart_disponents(df):
    # Count the occurrences of each type
    type_counts = df['disp'].value_counts()

    # Plot a pie chart
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of accounts with disponents')
    plt.show()

def pie_chart_loan_paid(df):
    # Count the occurrences of each status for each account_id
    status_distribution = df.groupby(['account_id', 'status']).size().unstack(fill_value=0)

    # Plotting the pie chart
    colors = ['lightcoral', 'lightblue']
    # explode = (0.1, 0)  # explode the 1st slice slightly

    plt.figure(figsize=(8, 8))
    plt.pie(status_distribution[1], labels=status_distribution.index, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Distribution of account_ids based on Status')
    plt.show()

def distribution_trans_balance(df_loan_trans_account):
    # Calculate the average amount and average balance for each account_id
    avg_amount_credit = df_loan_trans_account[df_loan_trans_account["type"] == 'credit'].groupby('account_id')['amount_trans'].mean()
    avg_amount_withdrawal = df_loan_trans_account[df_loan_trans_account["type"] == 'withdrawal'].groupby('account_id')['amount_trans'].mean()
    avg_balance = df_loan_trans_account.groupby('account_id')['balance'].mean()

    # Plotting the distribution of average amount in credit transactions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.hist(avg_amount_credit, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Average Amount in Credit Transactions')
    plt.xlabel('Average Amount')
    plt.ylabel('Frequency')

    # Plotting the distribution of average amount in credit transactions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 2)
    plt.hist(avg_amount_withdrawal, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Average Amount in Withdrawal Transactions')
    plt.xlabel('Average Amount')
    plt.ylabel('Frequency')

    # Plotting the distribution of average balance
    plt.subplot(1, 3, 3)
    plt.hist(avg_balance, bins=30, color='salmon', edgecolor='black')
    plt.title('Distribution of Average Balance')
    plt.xlabel('Average Balance')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------
# correlations
# ----------------------------------------------------------------

def pearson_correlation(df):
    mask = np.triu(df.corr(), 1)
    plt.figure(figsize=(19, 9))
    sns.heatmap(df.corr(), annot=True, vmax=1, vmin=-1, square=True, cmap='BrBG', mask=mask)
    plt.show()
    status_unpaid_correlation = pg.pairwise_corr(df, columns=['status'], method='pearson').loc[:,['X','Y','r']]
    status_unpaid_correlation.sort_values(by=['r'], ascending=False)
    print(status_unpaid_correlation)

def spearman_correlation(df):
    df_spear = df.copy()
    df_spear.drop(["status"], axis=1, inplace=True)

    spearman_rank = pg.pairwise_corr(df_spear, method='spearman').loc[:,['X','Y','r']]
    pos = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[:5,:]
    neg = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[-5:,:]
    con = pd.concat([pos,neg], axis=0)
    print(con.reset_index(drop=True))

    mask = np.triu(df_spear.corr(method='spearman'), 1)
    plt.figure(figsize=(19, 9))
    sns.heatmap(df_spear.corr(method='spearman'), annot=True, vmax=1, vmin=-1, square=True, cmap='BrBG', mask=mask);
    plt.show()

df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loan)
# plot_balance_graphs(df_trans_sorted, df_loans_sorted)
df_loan_trans_account = merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted)
accounts_with_disponent = accounts_with_disponents(df_loan_trans_account, df_disp)
df_loan_trans_account = add_disponent_info_to_loan_trans(df_loan_trans_account, accounts_with_disponent)
df_loan_trans_account = transform_status(df_loan_trans_account)

# simple_export(df_loan_trans_account, "new_db.csv")

# unbalanced data: 12.1% of entries with loan not paid
# pie_chart_loan_paid(df_loan_trans_account)

# pearson_correlation(df_loan_trans_account)

# based on this result, we can decide which 2 features to drop from [amount_loan, payments or duration]
# they are strongly correlated
# spearman_correlation(df_loan_trans_account)

# ----------------------------------------------------------------
# data cleaning
# ----------------------------------------------------------------

# outliers

def outliers(df):
    df_categorical = df.select_dtypes(exclude=["int64","float64"]).copy()
    df_numerical = df.select_dtypes(exclude=["object","category"]).copy()
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(16, 8))
    index = 0
    axs = axs.flatten()
    for k,v in df_numerical.items():
        sns.boxplot(y=k, data=df_numerical, ax=axs[index], orient="h")
        index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        plt.show()

# outliers(df_loan_trans_account)

# ----------------------------------------------------------------
#  feature selection
# ----------------------------------------------------------------

# exclude amount_trans, it has no predicitve value (p > 0.05)
# keep all categorical features
def feature_selection(df):
    # numerical
    df_numerical = df.select_dtypes(exclude=["object","category"]).copy()
    Xnum = df_numerical.drop(["status"], axis= "columns")
    ynum = df_numerical.status

    # drop ids, irrelevant
    df.drop(['account_id', 'loan_id'], axis=1, inplace=True)

    # p values
    print(pd.DataFrame(
        [scipy.stats.pearsonr(Xnum[col], 
        ynum) for col in Xnum.columns], 
        columns=["Pearson Corr.", "p-value"], 
        index=Xnum.columns,
    ).round(4))
    df.drop(['amount_trans'], axis=1, inplace=True)

    # categorical
    Xcat = df.select_dtypes(exclude=['int64','float64']).copy()
    Xcat['target'] = df.status
    Xcat.dropna(how="any", inplace=True)
    ycat = Xcat.target
    Xcat.drop("target", axis=1, inplace=True)

    print()

    # Chi-square test for independence
    for col in Xcat.columns:
        table = pd.crosstab(Xcat[col], ycat)
        print(table)
        print()
        _, pval, _, expected_table = scipy.stats.chi2_contingency(table)
        print(f"p-value: {pval:.25f}")
    return df

def encoding_categorical(df):
    df_type = df.type
    df_disponent = df.disp
    # type of transaction
    t_ohe = pd.get_dummies(df_type)
    # print(t_ohe)
    bin_enc_term = BinaryEncoder()
    t_bin = bin_enc_term.fit_transform(df_type)
    # print(t_bin)
    # with disponent or not
    d_ohe = pd.get_dummies(df_disponent)
    # print(d_ohe)
    bin_enc_home = BinaryEncoder()
    d_bin = bin_enc_home.fit_transform(df_disponent)
    # print(d_ohe)

    df = pd.get_dummies(df, columns=['type', 'disp'])

    # print(df)

    return df

df_loan_trans_account = feature_selection(df_loan_trans_account)
df_loan_trans_account = encoding_categorical(df_loan_trans_account)
simple_export(df_loan_trans_account, "new_db.csv")

# V2
# ----------------------------------------------------------------
# Predictive Modelling
# ----------------------------------------------------------------

# ROC Curve: Area Under the Curve
def auc_roc_plot(y_test, y_preds):
    fpr, tpr, thresholds = roc_curve(y_test,y_preds)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

seed = 42

def logistic_regression(df):
    train_set_lr, test_set_lr = train_test_split(df, test_size = 0.2, random_state = seed)
    X_train_lr = train_set_lr.drop(['status'], axis = 1)
    y_train_lr = train_set_lr['status']
    X_test_lr = test_set_lr.drop(['status'], axis = 1)
    y_test_lr = test_set_lr['status']

    # Normalizing the train and test data
    scaler_lr = MinMaxScaler()
    features_names = X_train_lr.columns
    X_train_lr = scaler_lr.fit_transform(X_train_lr)
    X_train_lr = pd.DataFrame(X_train_lr, columns = features_names)
    X_test_lr = scaler_lr.transform(X_test_lr)
    X_test_lr = pd.DataFrame(X_test_lr, columns = features_names)

    lr = LogisticRegression(max_iter = 1000, solver = 'lbfgs', random_state = seed, class_weight = 'balanced' )
    parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    clf_lr = GridSearchCV(lr, parameters, cv = 5).fit(X_train_lr, y_train_lr)

    y_preds_lr = clf_lr.predict_proba(X_test_lr)[:,1]

    auc_roc_plot(y_test_lr, y_preds_lr)

    # Confusion Matrix display
    tn, fp, fn, tp = confusion_matrix(y_test_lr == 1, y_preds_lr > 0.5).ravel()
    print(confusion_matrix(y_test_lr == 1, y_preds_lr > 0.5).ravel())

def knn(df):
    # dividing the dataset in train (80%) and test (20%)
    train_set_knn, test_set_knn = train_test_split(df, test_size = 0.2, random_state = seed)
    X_train_knn = train_set_knn.drop(['status'], axis = 1)
    y_train_knn = train_set_knn['status']
    X_test_knn = test_set_knn.drop(['status'], axis = 1)
    y_test_knn = test_set_knn['status']

    # normalizing train and test data
    scaler_knn = MinMaxScaler()
    features_names = X_train_knn.columns
    X_train_knn = scaler_knn.fit_transform(X_train_knn)
    X_train_knn = pd.DataFrame(X_train_knn, columns = features_names)
    X_test_knn = scaler_knn.transform(X_test_knn)
    X_test_knn = pd.DataFrame(X_test_knn, columns = features_names)

    # time
    best_k = 0
    max_acc = 0
    for k in range(1, 200, 5):
        k = k + 1
        knn = KNeighborsClassifier(n_neighbors = k).fit(X_train_knn, y_train_knn)
        acc = knn.score(X_test_knn, y_test_knn)
        if acc > max_acc:
            max_acc = acc
            best_k = k
        # print('Accuracy for k =', k, ' is:', acc)
    print('Accuracy for best k =', best_k, ' is:', max_acc)

    knn = KNeighborsClassifier(n_neighbors = best_k, weights='uniform').fit(X_train_knn, y_train_knn)
    y_preds_knn = knn.predict(X_test_knn)
    auc_roc_plot(y_test_knn, y_preds_knn)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_knn == 1, y_preds_knn > 0.5).ravel()
    print(confusion_matrix(y_test_knn == 1, y_preds_knn > 0.5).ravel())

knn(df_loan_trans_account)