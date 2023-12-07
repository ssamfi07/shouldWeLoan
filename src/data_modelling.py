import pandas as pd

import exploratory_plots
import utils
import correlations
import modelling_v1
import feature_engineering
import modelling_v2

# read the input data csv files
df_account = pd.read_csv('../bank/account.csv', sep=';', low_memory=False)
df_client = pd.read_csv('../bank/client.csv', sep=';', low_memory=False)
df_disp = pd.read_csv('../bank/disp.csv', sep=';', low_memory=False)
df_trans = pd.read_csv('../bank/trans_dev.csv', sep=';', low_memory=False)
df_loan = pd.read_csv('../bank/loan_dev.csv', sep=';', low_memory=False)
df_card = pd.read_csv('../bank/card_dev.csv', sep=';', low_memory=False)
df_district = pd.read_csv('../bank/district.csv', sep=';', low_memory=False)

df_trans_comp = pd.read_csv('../bank/trans_comp.csv', sep=';', low_memory=False)
df_loans_comp = pd.read_csv('../bank/loan_comp.csv', sep=';', low_memory=False)

# function used for the aggregated data from kaggle
def merge_and_export_aggregated(df_trans, df_loans, fileName):
    df_trans_sorted, df_loans_sorted = utils.sort_trans_loans_by_account_id(df_trans, df_loans)
    total = utils.aggregation(df_trans_sorted, df_loans_sorted)
    # drop account_id, we already have loan_id
    total.drop('account_id', axis=1, inplace=True)
    utils.simple_export(total, fileName)
    print(total)

def select_pred_account(account_id):
    df_predict_account = pd.read_csv('../csv_exports/account_id'+ str(account_id) + '.csv', sep=',', low_memory=False)
    return df_predict_account

df_trans_sorted, df_loans_sorted = utils.sort_trans_loans_by_account_id(df_trans, df_loan)
# data we want to predict the status for - it should have the correct features
df_predict_account_x = select_pred_account(6)

# drop irrelevant features
df_loan_trans_account = utils.merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted)
df_predict_account_x = utils.drop_features(df_predict_account_x)
# handle missing values in column date_diff for the prediction db
df_predict_account_x['date_diff'].fillna(0, inplace=True)

# find the disponent information
accounts_with_disponent = utils.accounts_with_disponents(df_loan_trans_account, df_disp)
new_accounts_with_disponent = utils.accounts_with_disponents(df_predict_account_x, df_disp)

# add disponent info
df_loan_trans_account = feature_engineering.add_disponent_info_to_loan_trans(df_loan_trans_account, accounts_with_disponent)
df_predict_account_x = feature_engineering.add_disponent_info_to_loan_trans(df_predict_account_x, new_accounts_with_disponent)

# for predicted we don't need because we want to fill it
# but we can create the column
df_loan_trans_account = feature_engineering.transform_status_to_binary(df_loan_trans_account)

# simple_export(df_loan_trans_account, "new_db.csv")

# ----------------------------------------------------------------
# Exploratory data visualization
# ----------------------------------------------------------------

# each account transactions and balance evolution -- commented because we have a lot of accounts
# exploratory_plots.plot_balance_graphs(df_trans_sorted, df_loans_sorted)

# accounts with disponents proportions
# exploratory_plots.pie_chart_disponents(df_loan_trans_account)

# paid vs unpaid loans
# unbalanced data: 14% of entries with loan not paid
# exploratory_plots.pie_chart_loan_paid(df_loan_trans_account)

# distribution of transactions and balance
# exploratory_plots.distribution_trans_balance(df_loan_trans_account)

# stats
# exploratory_plots.stats("balance", df_loan_trans_account)

# scatters
# exploratory_plots.scatter("amount_loan", "amount_trans", df_loan_trans_account)

# barh
# exploratory_plots.bar("amount_trans", df_loan_trans_account)

# ----------------------------------------------------------------
# Correlations
# ----------------------------------------------------------------

# correlations between the features and the target
# correlations.pearson_correlation(df_loan_trans_account)

# correlations between each feature
# based on this result, we can decide which 2 features to drop from [amount_loan, payments or duration]
# they are strongly correlated
# correlations.spearman_correlation(df_loan_trans_account)

# ----------------------------------------------------------------
# feature selection
# ----------------------------------------------------------------
# feature_engineering.outliers(df_loan_trans_account)
df_loan_trans_account = feature_engineering.feature_selection(df_loan_trans_account)
df_loan_trans_account = feature_engineering.encoding_categorical(df_loan_trans_account)

# encode also the database for prediction
df_predict_account_x = feature_engineering.feature_selection(df_predict_account_x)
df_predict_account_x = feature_engineering.encoding_categorical(df_predict_account_x)

utils.simple_export(df_loan_trans_account, "new_db.csv")
utils.simple_export(df_predict_account_x, "new_db_predicted.csv")

# ----------------------------------------------------------------
# modelling and evaluation
# ----------------------------------------------------------------

trained_lr_model, scaler = modelling_v2.logistic_regression(df_loan_trans_account)
predictions = modelling_v2.predict_status_lr(trained_lr_model, df_predict_account_x)
print(predictions)
# modelling_v2.knn(df_loan_trans_account)