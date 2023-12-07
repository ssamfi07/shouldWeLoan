import pandas as pd
import os

# ---------------------------------------------------------------------------
# utility functions for sorting, merging, filtering and aggregating features
# ---------------------------------------------------------------------------

def sort_trans_loans_by_account_id(df_trans, df_loan=None):
    # sort by account and date
    df_trans_sorted = df_trans.sort_values(by=["account_id", "date"])
    # print(df_trans_sorted[df_trans_sorted['account_id'] == 19])

    df_loans_sorted = None

    # sort by account and date
    if df_loan is not None:
        df_loans_sorted = df_loan.sort_values(by=["account_id", "date"])

    # Format the date to epoch time so it's easier to work with later
    df_trans_sorted['date'] = pd.to_datetime(df_trans_sorted['date'], format='%y%m%d').astype(int) // 10**9
    # print(df_trans_sorted)

    # Calculate the date difference for each transaction type within each account
    df_trans_sorted['date_diff'] = df_trans_sorted.groupby(['account_id', 'type'])['date'].diff()

    # handle missing values in date_diff
    df_trans_sorted['date_diff'].fillna(0, inplace=True)

    return df_trans_sorted, df_loans_sorted

# for the original database.csv
def export_trans_loans_merge(df_trans, df_loan, file_name):
    df_trans_sorted, df_loans_sorted = sort_trans_loans_by_account_id(df_trans, df_loan)
    merged = pd.merge(df_loans_sorted, df_trans_sorted, on='account_id', how='left')
    merged.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id', 'account_id'], axis=1, inplace=True)
    merged.rename(columns={'amount_x': 'amount_loan', 'amount_y': 'amount_trans'}, inplace=True)
    merged.to_csv(file_name, sep=',', index=False, encoding='utf-8')

def drop_features(df):
    # we will drop some columns -- date_x corresponds to the issuing date of the loan, it is always after the transaction history ends
    if 'date_x' and 'date_y' in df.columns:
        df.drop(['bank', 'account', 'operation', 'k_symbol', 'date_x', 'date_y', 'trans_id'], axis=1, inplace=True)
    elif 'date' in df.columns:
        df.drop(['bank', 'account', 'operation', 'k_symbol', 'date', 'trans_id'], axis=1, inplace=True)
    return df

def merge_trans_loans_and_drop_features(df_loans_sorted, df_trans_sorted):
    # rename columns
    df_loans_sorted.rename(columns={'amount': 'amount_loan'}, inplace=True)
    df_trans_sorted.rename(columns={'amount': 'amount_trans'}, inplace=True)
    # merge transactions with loans based on account id
    df_loan_trans_account = pd.merge(df_loans_sorted, df_trans_sorted, on='account_id', how='inner')
    # we will drop some columns -- date_x corresponds to the issuing date of the loan, it is always after the transaction history ends
    drop_features(df_loan_trans_account)

    # Replace "withdrawal in cash" with "withdrawal" for all types
    df_loan_trans_account['type'] = df_loan_trans_account['type'].replace('withdrawal in cash', 'withdrawal')

    # maybe for amount of transaction, if the type is withdrawal we need to set the amount to the opposite value (negative)
    df_loan_trans_account['amount_trans'] = df_loan_trans_account.apply(
        lambda row: -row['amount_trans'] if row['type'] == 'withdrawal' else row['amount_trans'],
        axis=1)

    return df_loan_trans_account

def simple_export(df, fileName):
    # get the current working directory
    current_dir = os.getcwd()
    # define the path to the csv_exports folder
    exports_folder = os.path.join(current_dir, '..', 'csv_exports')
    # if the folder doesn't exist, create it
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)
    # define the full path to the CSV file
    full_path = os.path.join(exports_folder, fileName)
    df.to_csv(full_path, sep=',', index=False, encoding='utf-8')

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
    accounts_with_disponent = df_disp_filtered[df_disp_filtered['type'] == 'DISPONENT']['account_id'].tolist()
    return accounts_with_disponent
