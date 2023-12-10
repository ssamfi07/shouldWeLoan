import modelling_v2
import pandas as pd

df_improved = pd.read_csv('../csv_exports/improved_db.csv', sep=',', low_memory=False)

modelling_v2.NN(df_improved)