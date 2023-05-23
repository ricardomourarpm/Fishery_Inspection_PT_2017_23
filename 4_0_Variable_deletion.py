import pandas as pd
import numpy as np

#read fiscrep full
df = pd.read_pickle(r'data\final_fiscrep_selected.pickle')

import pandas as pd

columns = df.columns.tolist()

# read CFR_df
CFR_df = pd.read_pickle('data\CFR_data.pickle')

# direct identifiers dropped
df = df.drop(columns=['Reg_Num', 'Registration Number_y'])

columns = df.columns.tolist()

# key variables dropped

df = df.drop(columns=['Name'])

columns = df.columns.tolist()

# not obvious key variables to drop


for variable in columns:
    value_counts = df[variable].value_counts()
    value_counts_one = value_counts[value_counts==1]
    print(value_counts_one)

columns_CFR = CFR_df.columns.tolist()

for variable in columns_CFR:
    value_counts = CFR_df[variable].value_counts()
    value_counts_one = value_counts[value_counts==1]
    print(value_counts_one)

# Drop GDH
df = df.drop(columns=['GDH'])

# Zero values are in fact nan
df[['LOA', 'LBP', 'Tonnage GT', 'Other tonnage', 'Power of main engine', 'Power of auxiliary engine']
   ] = df[['LOA', 'LBP', 'Tonnage GT', 'Other tonnage', 'Power of main engine', 'Power of auxiliary engine']].replace(0, np.nan)

#save fiscrep
df.to_pickle(r'data\final_fiscrep_deleted.pickle')
df.to_csv(r'data\final_fiscrep_deleted.csv')
