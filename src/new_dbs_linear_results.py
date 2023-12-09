import pandas as pd
import modelling_v2

df_pcs = pd.read_csv('../bank/dfs_pcs.csv', sep=',', low_memory=False)
df_new_db2 = pd.read_csv('../bank/new_db2.csv', sep=',', low_memory=False)
df_pcs_only = pd.read_csv('../bank/pcs_only.csv', sep=',', low_memory=False)

trained_lr_model1, scaler = modelling_v2.logistic_regression(df_pcs)
trained_lr_model2, scaler = modelling_v2.logistic_regression(df_new_db2)
trained_lr_model3, scaler = modelling_v2.logistic_regression(df_pcs_only)
