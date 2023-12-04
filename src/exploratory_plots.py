import pandas as pd
import matplotlib.pyplot as plt

def plot_balance_graphs(df_trans_sorted, df_loans_sorted):
    # for all accounts make the balance graph in time based in transactions - deposits(lines) or expenditures(dots)
    for account in range(1,10000):
        df_account = df_trans_sorted[df_trans_sorted['account_id'] == account]
        if not df_account.empty:
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
    # drop duplicate rows with regards to loan_id
    df = df.drop_duplicates(subset='loan_id', keep='first')
    # count the occurrences of each status for each loan_id
    status_distribution = df.groupby(['status']).size()

    # Plotting the pie chart
    colors = ['lightcoral', 'lightblue']
    plt.figure(figsize=(8, 8))
    plt.pie(status_distribution, labels=status_distribution.index, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Distribution of loan_ids based on Status')
    plt.show()

def distribution_trans_balance(df_loan_trans_account):
    # Calculate the average amount and average balance for each account_id
    avg_amount_credit = df_loan_trans_account[df_loan_trans_account["type"] == 'credit'].groupby('account_id')['amount_trans'].mean()
    avg_amount_withdrawal = df_loan_trans_account[df_loan_trans_account["type"] == 'withdrawal'].groupby('account_id')['amount_trans'].mean()
    avg_balance = df_loan_trans_account.groupby('account_id')['balance'].mean()

    # Plotting the distribution of average amount in credit transactions
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.hist(avg_amount_credit, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Average Amount in Credit Transactions')
    plt.xlabel('Average Amount')
    plt.ylabel('Frequency')

    # Plotting the distribution of average amount in credit transactions
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