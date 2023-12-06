import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

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

# General statistics
def stats(x, data):
    print(f"Variable: {x}")
    print(f"Type of variable: {data[x].dtype}")
    print(f"Total observations: {data[x].shape[0]}")
    detect_null_val = data[x].isnull().values.any()
    if detect_null_val:
        print(f"Missing values: {data[x].isnull().sum()} ({(data[x].isnull().sum() / data[x].isnull().shape[0] *100).round(2)}%)")
    else:
        print(f"Missing values? {data[x].isnull().values.any()}")
    print(f"Unique values: {data[x].nunique()}")
    if data[x].dtype != "O":
        print(f"Min: {int(data[x].min())}")
        print(f"25%: {int(data[x].quantile(q=[.25]).iloc[-1])}")
        print(f"Median: {int(data[x].median())}")
        print(f"75%: {int(data[x].quantile(q=[.75]).iloc[-1])}")
        print(f"Max: {int(data[x].max())}")
        print(f"Mean: {data[x].mean()}")
        print(f"Std dev: {data[x].std()}")
        print(f"Variance: {data[x].var()}")
        print(f"Skewness: {scipy.stats.skew(data[x])}")
        print(f"Kurtosis: {scipy.stats.kurtosis(data[x])}")
        print("")

        # Percentiles 1%, 5%, 95% and 99%
        print("Percentiles 1%, 5%, 95%, 99%")
        print(data[x].quantile(q=[.01, .05, .95, .99]))
        print("")
    else:
        print(f"List of unique values: {data[x].unique()}")

# Variable vs. target chart
def target(x, data):
    short_0 = data[data.status == 0].loc[:,x]
    short_1 = data[data.status == 1].loc[:,x]

    a = np.array(short_0)
    b = np.array(short_1)

    np.warnings.filterwarnings('ignore')

    plt.hist(a, bins=40, density=True, color="g", alpha = 0.6, label='Not-default', align="left")
    plt.hist(b, bins=40, density=True, color="r", alpha = 0.6, label='Default', align="right")
    plt.legend(loc='upper right')
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()

# Boxplot + Hist chart
def boxhist(x, data):
    variable = data[x]
    np.array(variable).mean()
    np.median(variable)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.5, 2)})
    mean=np.array(variable).mean()
    median=np.median(variable)
    sns.boxplot(variable, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    sns.distplot(variable, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    plt.title(x, fontsize=10, loc="right")
    plt.legend({'Mean':mean,'Median':median})
    ax_box.set(xlabel='')
    plt.show()

# histogram
def hist(x, data):
    plt.hist(data[x], bins=25)
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()

# pie chart
def pie(x, data):
    data[x].value_counts(dropna=False).plot(kind='pie', figsize=(6,5), fontsize=10, autopct='%1.1f%%', startangle=0, legend=True, textprops={'color':"white", 'weight':'bold'});
    # Number of observations by class
    obs = data[x].value_counts(dropna=False)
    o = pd.DataFrame(obs)
    o.rename(columns={x:"Freq abs"}, inplace=True)
    o_pc = (data[x].value_counts(normalize=True) * 100).round(2)
    obs_pc = pd.DataFrame(o_pc)
    obs_pc.rename(columns={x:"percent %"}, inplace=True)
    obs = pd.concat([o,obs_pc], axis=1)
    print(obs)

# Variable vs. target chart
def target(x, data):
    short_0 = data[data.status == 0].loc[:,x]
    short_1 = data[data.status == 1].loc[:,x]

    a = np.array(short_0)
    b = np.array(short_1)

    np.warnings.filterwarnings('ignore')

    plt.hist(a, bins=40, density=True, color="g", alpha = 0.6, label='Not-default', align="left")
    plt.hist(b, bins=40, density=True, color="r", alpha = 0.6, label='Default', align="right")
    plt.legend(loc='upper right')
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()

# boxplot + Hist chart
def boxhist(x, data):
    variable = data[x]
    np.array(variable).mean()
    np.median(variable)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.5, 2)})
    mean=np.array(variable).mean()
    median=np.median(variable)
    sns.boxplot(variable, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    sns.distplot(variable, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    plt.title(x, fontsize=10, loc="right")
    plt.legend({'Mean':mean,'Median':median})
    ax_box.set(xlabel='')
    plt.show()

# bar chart
def bar(x, data):
    ax = data[x].value_counts().plot(kind="bar", figsize=(6,5), fontsize=10, color=sns.color_palette("rocket"), table=False)
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.xlabel(x, fontsize=10)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.ylabel("Absolute values", fontsize=10)
    plt.title(x, fontsize=10, loc="right")
    plt.show()

# barh chart
def barh(x, data):
    data[x].value_counts().plot(kind="barh", figsize=(6,5), fontsize=10, color=sns.color_palette("rocket"), table=False)
    plt.xlabel("Absolute values", fontsize=10)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.ylabel(x, fontsize=10)
    plt.title(x, fontsize=10, loc="right")
    plt.show()

# pivot_table_mean
def pivot_mean(a, b, c, data):
    type_pivot_mean = data.pivot_table(
        columns=a,
        index=b,
        values=c, aggfunc=np.mean)
    print(type_pivot_mean)
    # display pivot_table
    type_pivot_mean.sort_values(by=[b], ascending=True).plot(kind="bar", title=(b), figsize=(6,4),fontsize = 12);

# pivot_table_sum
def pivot_sum(a, b, c, data):
    type_pivot_sum = data.pivot_table(
        columns=a,
        index=b,
        values=c, aggfunc=np.sum)
    print(type_pivot_sum)
    # display pivot_table
    type_pivot_sum.sort_values(by=[b], ascending=True).plot(kind="bar", title=(b), figsize=(6,4),fontsize = 12);

# scatter plot
def scatter(x, y, data):
    targets = data["status"].unique()
    for target in targets:
            a = data[data["status"] == target][x]
            b = data[data["status"] == target][y]
    plt.scatter(a, b, label=f"status: {target}", marker="*")
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.title("abc", fontsize=10, loc="right")
    plt.legend()
    plt.show()
