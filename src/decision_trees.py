import modelling_v2
import pandas as pd

df_improved = pd.read_csv('../csv_exports/improved_db.csv', sep=',', low_memory=False)

modelling_v2.decision_trees(df_improved)
