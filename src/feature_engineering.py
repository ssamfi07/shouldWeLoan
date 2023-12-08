
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder

import scipy

# ----------------------------------------------------------------
# feature engineering
# ----------------------------------------------------------------

# returns a DataFrame with means and stds for balance and amounts, and the no of transactions
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
    if 'status' in df_loan_trans_account.columns:
        unique_loan_info = df_loan_trans_account[['loan_id', 'amount_loan', 'duration', 'status']].drop_duplicates()
    else: # for data we want tp predict, status is missing
        unique_loan_info = df_loan_trans_account[['loan_id', 'amount_loan', 'duration']].drop_duplicates()
    # Merge loan_data with unique_loan_info based on loan_id
    loan_data = pd.merge(loan_data, unique_loan_info, on='loan_id', how='left')

    return loan_data

# 1 for account with disponent, 0 for account without disponent
def add_disponent_info_to_loan_trans(df_loan_trans_account, accounts_with_disponent):
    df_loan_trans_account['disp'] = df_loan_trans_account.apply(
        lambda row: "disponent" if row['account_id'] in accounts_with_disponent else "no_disponent",
        axis=1)
    return df_loan_trans_account

# status transformation: 0 for -1 and 1 for 1
def transform_status_to_binary(df_loan_trans_account):
    df_loan_trans_account['status'] = df_loan_trans_account.apply(
        lambda row: 1 if row['status'] == 1 else 0,
        axis=1)
    return df_loan_trans_account

def transform_disponent_to_binary(df_loan_trans_account):
    df_loan_trans_account['disp'] = df_loan_trans_account.apply(
        lambda row: 1 if row['disp'] == "disponent" else 0,
        axis=1)
    return df_loan_trans_account

# type and disp columns one hot encoded
def encoding_categorical(df):
    if 'type' in df.columns:
        df_type = df.type
        # type of transaction
        t_ohe = pd.get_dummies(df_type)
        # print(t_ohe)
        bin_enc_term = BinaryEncoder()
        t_bin = bin_enc_term.fit_transform(df_type)
        df = pd.get_dummies(df, columns=['type'])

    if 'disp' in df.columns:
        # df_disponent = df.disp
        # with disponent or not
        # d_ohe = pd.get_dummies(df_disponent)
        # bin_enc_home = BinaryEncoder()
        # d_bin = bin_enc_home.fit_transform(df_disponent)
        # print(d_ohe)
        # df['disp'] = pd.get_dummies(df, columns=['disp'])
        transform_disponent_to_binary(df)

    # for the predicted df, we check if a column doesn't exist and we fill it with zeros
    if 'disp_no_disponent' not in df.columns:
        df['disp_no_disponent'] = 0
    elif 'disp_disponent' not in df.columns:
        df['disp_disponent'] = 0
    # print(df)

    return df

# outliers
# def outliers(df):
#     df_categorical = df.select_dtypes(exclude=["int64","float64"]).copy()
#     df_numerical = df.select_dtypes(exclude=["object","category"]).copy()
#     fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(16, 8))
#     index = 0
#     axs = axs.flatten()
#     for k,v in df_numerical.items():
#         sns.boxplot(y=k, data=df_numerical, ax=axs[index], orient="h")
#         index += 1
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#         plt.show()

# ----------------------------------------------------------------
#  feature selection
# ----------------------------------------------------------------

# exclude amount_trans, it has no predicitve value (p > 0.05)
# keep all categorical features
def feature_selection(df):
    # numerical
    df_numerical = df.select_dtypes(exclude=["object","category"]).copy()
    Xnum = df_numerical
    if 'status' in df_numerical.columns:
        Xnum = df_numerical.drop(["status"], axis= "columns")
        ynum = df_numerical.status
        print("P values for numerical features:")
        print()

        # p values
        print(pd.DataFrame(
            [scipy.stats.pearsonr(Xnum[col], 
            ynum) for col in Xnum.columns], 
            columns=["Pearson Corr.", "p-value"], 
            index=Xnum.columns,
        ).round(4))

        # categorical
        Xcat = df.select_dtypes(exclude=['int64','float64']).copy()
        Xcat['target'] = df.status
        Xcat.dropna(how="any", inplace=True)
        ycat = Xcat.target
        Xcat.drop("target", axis=1, inplace=True)

        print()
        print("P values for categorical features:")
        print()

        # Chi-square test for independence
        for col in Xcat.columns:
            table = pd.crosstab(Xcat[col], ycat)
            print(table)
            print()
            _, pval, _, expected_table = scipy.stats.chi2_contingency(table)
            print(f"p-value: {pval:.25f}")
        
    # drop ids, irrelevant
    df.drop(['account_id'], axis=1, inplace=True)

    # drop amount_trans based on the p-value evaluation for numerical features
    if 'amount' in df.columns:
        df.drop(['amount_trans'], axis=1, inplace=True)
    return df