import pandas as pd

import utils
import feature_engineering
import modelling_v2

df_trans_comp = pd.read_csv('../bank/trans_comp.csv', sep=';', low_memory=False)
df_loans_comp = pd.read_csv('../bank/loan_comp.csv', sep=';', low_memory=False)
df_improved = pd.read_csv('../csv_exports/new_db.csv', sep=',', low_memory=False)

df_trans_sorted, df_loans_sorted = utils.sort_trans_loans_by_account_id(df_trans_comp, df_loans_comp)

# drop irrelevant features
df_loan_trans_account = utils.merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted)

# aggregate
df_loan_trans_account = feature_engineering.aggregation(df_loan_trans_account)

# drop status column
df_loan_trans_account.drop(['status'], axis=1, inplace=True)

# remember the loan_ids
loan_ids = df_loan_trans_account['loan_id']

print(loan_ids)

df_loan_trans_account = feature_engineering.feature_selection(df_loan_trans_account)
df_loan_trans_account = feature_engineering.encoding_categorical(df_loan_trans_account)

utils.simple_export(df_loan_trans_account, "kaggle_db.csv")

# we need to drop the disp column from the improved df to work with the kaggle data
df_improved.drop(['disp'], axis=1, inplace=True)

# train model on the improved df
model, scaler = modelling_v2.logistic_regression(df_improved)

# Standardize the features for the predicted df
X_scaled = scaler.transform(df_loan_trans_account)

# Predict probabilities on the new dataset
probabilities = model.predict_proba(X_scaled)[:, 0]  # Probability of status being -1

# Prepare a DataFrame with loan_id and predicted probabilities
result_df = pd.DataFrame({'Id': loan_ids, 'Predicted': probabilities})

utils.simple_export(result_df, "kaggle_result.csv")
