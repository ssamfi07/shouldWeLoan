import pandas as pd

import exploratory_plots
import utils

# read the input data csv files with transactions
df_trans = pd.read_csv('../bank/trans_dev.csv', sep=';', low_memory=False)

# !!! make sure that the account_ids are valid before trying to export
# !!! because the accounts have to match the amount_loan, duration and payments for the loan
# loan_info structure:
# loan_id it's a list of lists
# account_id (identifier), loan_id, amount_loan, duration, payments
# the amount_loan needs to barely surpass a threshold which enables the prediction of status 0
loan_info = {
    1: [12345, 120000, 24, 3333,3333],
    4: [5678, 150000, 24, 1250],
    6: [91011, 210000, 24, 1666,66]
}

# identifiers for loan_info
account_ids = [1, 4, 6]

def create_testable_dbs_based_on_account_ids_and_loan_info(df, account_ids, loan_info, file_name):
    df_trans_sorted, _ = utils.sort_trans_loans_by_account_id(df)
    # filter based on the account_ids
    filtered_df = df_trans_sorted[df_trans_sorted['account_id'].isin(account_ids)]
    filtered_df.rename(columns={'amount': 'amount_trans'}, inplace=True)
    # add for each row the corresponding loan_id, amount_loan, duration and payments
    # add loan_id, amount_loan, duration and payments for each row
    for index, row in filtered_df.iterrows():
        account_id = row['account_id']
        if account_id in account_ids:
            loan_info_values = loan_info.get(account_id, [])
            filtered_df.loc[index, 'loan_id'] = loan_info_values[0]
            filtered_df.loc[index, 'amount_loan'] = loan_info_values[1]
            filtered_df.loc[index, 'duration'] = loan_info_values[2]
            filtered_df.loc[index, 'payments'] = loan_info_values[3]
    # export each unique account_id to a separate CSV file
    for account_id in account_ids:
        account_df = filtered_df[filtered_df['account_id'] == account_id]
        file_path = f"{file_name}_id{account_id}.csv"
        utils.simple_export(account_df, file_path)
        print(f"Exported DataFrame for account_id {account_id} to {file_path}")

create_testable_dbs_based_on_account_ids_and_loan_info(df_trans, account_ids, loan_info, "account")
