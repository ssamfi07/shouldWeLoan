import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pingouin as pg

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