import pandas as pd

import exploratory_plots
import utils
import correlations
import modelling_v1
import feature_engineering
import modelling_v2
import testable_samples

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

# drop irrelevant features
df_loan_trans_account = utils.merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted)

# find the disponent information
accounts_with_disponent = utils.accounts_with_disponents(df_loan_trans_account, df_disp)

df_loan_trans_account = feature_engineering.transform_status_to_binary(df_loan_trans_account)

# simple_export(df_loan_trans_account, "new_db.csv")

# ----------------------------------------------------------------
# Exploratory data visualization
# ----------------------------------------------------------------

# each account transactions and balance evolution -- commented because we have a lot of accounts
# df_accounts_with_no_loan = exploratory_plots.plot_balance_graphs(df_trans_sorted, df_loans_sorted)
# utils.simple_export(df_accounts_with_no_loan, "accounts_with_no_loan.csv")

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
# feature selection
# ----------------------------------------------------------------
# feature_engineering.outliers(df_loan_trans_account)

# aggregate
df_loan_trans_account = feature_engineering.aggregation(df_loan_trans_account)

# add disponent info
df_loan_trans_account = feature_engineering.add_disponent_info_to_loan_trans(df_loan_trans_account, accounts_with_disponent)

# ----------------------------------------------------------------
# Correlations
# ----------------------------------------------------------------

# correlations between the features and the target
# correlations.pearson_correlation(df_loan_trans_account)

# correlations between each feature
# based on this result, we can decide which 2 features to drop from [amount_loan, payments or duration]
# they are strongly correlated
# correlations.spearman_correlation(df_loan_trans_account)

df_loan_trans_account = feature_engineering.feature_selection(df_loan_trans_account)
df_loan_trans_account = feature_engineering.encoding_categorical(df_loan_trans_account)

utils.simple_export(df_loan_trans_account, "new_db.csv")

# ----------------------------------------------------------------
# modelling and evaluation
# ----------------------------------------------------------------

trained_lr_model, scaler = modelling_v2.logistic_regression(df_loan_trans_account)
# modelling_v2.knn(df_loan_trans_account)

# ----------------------------------------------------------------
# process prediction datasets and make the predictions
# ----------------------------------------------------------------

# !!! individual accounts from the dictionary
"""
for predict_account in testable_samples.loan_info:
    print(predict_account)
    df_predict = select_pred_account(predict_account)
    # data we want to predict the status for - it should have the correct features
    df_predict_account[predict_account] = df_predict
    # print(df_predict_account[predict_account])
    # drop features
    df_predict_account[predict_account] = utils.drop_features_and_rename(df_predict_account[predict_account])
    # aggregation
    df_predict_account[predict_account] = feature_engineering.aggregation(df_predict_account[predict_account])
    # find and add disponent info
    new_accounts_with_disponent = utils.accounts_with_disponents(df_predict_account[predict_account], df_disp)
    df_predict_account[predict_account] = feature_engineering.add_disponent_info_to_loan_trans(df_predict_account[predict_account], new_accounts_with_disponent)
    # encode also the database for prediction
    df_predict_account[predict_account] = feature_engineering.feature_selection(df_predict_account[predict_account])
    df_predict_account[predict_account] = feature_engineering.encoding_categorical(df_predict_account[predict_account])
    # export to unique csv files
    utils.simple_export(df_predict_account[predict_account], "new_db_predicted" + str(predict_account) + ".csv")
    # predict and print
    predictions = modelling_v2.predict_status_lr(trained_lr_model, df_predict_account[predict_account])
    print(predictions)
"""

# ----------------------------------------------------------------
# process the accounts without loan
# ----------------------------------------------------------------

accounts_without_loans = utils.account_without_loan(df_trans_sorted, df_loans_sorted)
accounts_without_loans = utils.drop_features_and_rename(accounts_without_loans)

# aggregate accounts without loan
accounts_without_loans = feature_engineering.aggregation(accounts_without_loans, no_loan_id=1)

# !!! these are outliers which affect our predictions
# filter out rows with num_transactions less than 10
accounts_without_loans = accounts_without_loans[accounts_without_loans['num_transactions'] >= 10]
# filter out rows with less than 3 transactions per month
accounts_without_loans = accounts_without_loans[(accounts_without_loans['num_transactions'] / accounts_without_loans['date_diff']) >= 3]

loan_info = testable_samples.calculate_amount_loan_and_duration(accounts_without_loans)
accounts_without_loans = testable_samples.add_amount_loan_and_duration_to_df(accounts_without_loans, loan_info)

# filter out rows with negative amount_loan -- formula works for balanced transactions
accounts_without_loans = accounts_without_loans[(accounts_without_loans['amount_loan'] >= 0)]

# disponents addition
accounts_without_loans = feature_engineering.add_disponent_info_to_loan_trans(accounts_without_loans, accounts_with_disponent)

# feature selection and encoding
accounts_without_loans = feature_engineering.feature_selection(accounts_without_loans)
accounts_without_loans = feature_engineering.encoding_categorical(accounts_without_loans)

# ----------------------------------------------------------------
# predictions for the accounts_without_loans
# ----------------------------------------------------------------

predictions = modelling_v2.predict_status_lr(trained_lr_model, accounts_without_loans)
print(predictions)

predictions_rounded = predictions.round().astype(int)

print(len(predictions_rounded))
print(len(accounts_without_loans))

# check and correct the length mismatch
if len(predictions_rounded) != len(accounts_without_loans):
    # adjust the length of predictions_rounded to match the DataFrame
    predictions_rounded = predictions_rounded[:len(accounts_without_loans)]

# add the predictions to the 'status' column for accounts_without_loans and export
accounts_without_loans['status'] = predictions_rounded

# reset the index
accounts_without_loans = accounts_without_loans.reset_index(drop=True)

utils.simple_export(accounts_without_loans, "accounts_with_no_loans.csv")

# ----------------------------------------------------------------
# training again with the improved data
# ----------------------------------------------------------------

improved_training_data = pd.concat([df_loan_trans_account, accounts_without_loans], ignore_index=True)
utils.simple_export(improved_training_data, "improved_db.csv")
exploratory_plots.pie_chart_loan_paid(improved_training_data)
trained_lr_model, scaler = modelling_v2.logistic_regression(improved_training_data)