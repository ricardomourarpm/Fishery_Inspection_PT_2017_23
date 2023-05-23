import pandas as pd
import numpy as np

# read fiscrep
df = pd.read_pickle(r'data\final_fiscrep_deleted.pickle')

# save original
original_df = df.copy()

# load CFR data
CFR_data = pd.read_pickle(r'data\CFR_data.pickle')

# Rename column
df.rename(columns = {'Vessel_Type_y':'Vessel Type'}, inplace = True)

# save last records
last_cfr_data = CFR_data[~CFR_data.duplicated(subset='CFR', keep='last')]

# Only keep Year for all

last_cfr_data['Date of entry into service'] = last_cfr_data['Date of entry into service'].apply(lambda x: int(x.year) if not pd.isnull(x) else np.nan)

last_cfr_data['Year of construction'] = last_cfr_data['Year of construction'].apply(lambda x: int(x.year) if not pd.isnull(x) else np.nan)

# matched columns in both df and CFR
df_cols = list(df.columns)
CFR_data_cols = list(CFR_data.columns)
common_cols = [col for col in df_cols if col in CFR_data_cols]

df_match = df[common_cols]
CFR_match = last_cfr_data[common_cols+['CFR']]

#df['Date of entry into service'] = df['Date of entry into service'].apply(lambda x: np.round(x.year/10)*10 if not pd.isnull(x) else np.nan)

#df['Year of construction'] = df['Year of construction'].apply(lambda x: np.round(x.year/10)*10 if not pd.isnull(x) else np.nan)

seed_r = 100

np.random.seed(seed_r)
alpha = np.random.uniform(0.2,0.4)

sigma_date = np.std([x.year for x in df['Date of entry into service'] if not pd.isnull(x)])

sigma_year = np.std([x.year for x in df['Year of construction'] if not pd.isnull(x)])

np.random.seed(seed_r)
df['Date of entry into service'] = df['Date of entry into service'].apply(lambda x: np.round(np.random.normal(loc=x.year, scale=alpha*sigma_date),0) if not pd.isnull(x) else np.nan)
np.random.seed(100)
df['Year of construction'] = df['Year of construction'].apply(lambda x: np.round(np.random.normal(loc=x.year, scale=alpha*sigma_year),0) if not pd.isnull(x) else np.nan)

# Plot difference

import matplotlib.pyplot as plt

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Date of Entry into Service')
ax.set_xlabel('Date of entry into service (Original)')
ax.set_ylabel('Date of Entry into Service (modified)')

# Customize the scatter plot
ax.scatter(df['Date of entry into service'], original_df['Date of entry into service'], s=10, c='green', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()


#Plot difference

diff_year = original_df['Year of construction']-pd.to_datetime(df['Year of construction'], format='%Y')

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Year of Construction')
ax.set_xlabel('Year of Construction (Original)')
ax.set_ylabel('Year of Construction (modified)')

# Customize the scatter plot
ax.scatter(original_df['Year of construction'], df['Year of construction'], s=10, c='blue', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()


df.rename(columns = {'matched_CFR':'CFR', 'Vessel_Type_y':'Vessel_Type'}, inplace = True)

#df[['LOA', 'LBP']] = df[['LOA', 'LBP']].round()

#CFR_match[['LOA', 'LBP']] = CFR_match[['LOA', 'LBP']].round()

sigma_LOA = df['LOA'].std()

sigma_LBP = df['LBP'].std()
np.random.seed(seed_r)
df['LOA']=df['LOA'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_LOA),1) if not pd.isnull(x) else np.nan)
np.random.seed(seed_r)
df['LBP']=df['LBP'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_LBP),1) if not pd.isnull(x) else np.nan)

df['LOA'] = [loa if loa>0 else original_df.LOA.min() if loa==0 else np.abs(loa) for loa in df['LOA']]

df['LBP'] = [lbp if lbp>0 else original_df.LBP.min() if lbp==0 else np.abs(lbp) for lbp in df['LOA']]

df['Power of auxiliary engine'] = (df['Power of auxiliary engine']/10).round()*10
df['Other tonnage'] = df['Other tonnage'].round()

#CFR_match['Other tonnage'] = (CFR_match['Other tonnage']).round()
#CFR_match['Power of auxiliary engine']  = (CFR_match['Power of auxiliary engine']/10).round()*10

sigma_power = df['Power of main engine'].std()
np.random.seed(seed_r)
df['Power of main engine'] = df['Power of main engine'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_power),1) if not pd.isnull(x) else np.nan)

df['Power of main engine'] = [power if power>0 else original_df['Power of main engine'].min() if power==0 else np.abs(power) for power in df['Power of main engine']]

sigma_ton = df['Tonnage GT'].std()
np.random.seed(seed_r)
df['Tonnage GT'] = df['Tonnage GT'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_ton),1) if not pd.isnull(x) else np.nan)

df['Tonnage GT'] = [ton if ton>0 else original_df['Tonnage GT'].min() if ton==0 else np.abs(ton) for ton in df['Tonnage GT']]

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Power of main engine')
ax.set_xlabel('Power of main engine (Original)')
ax.set_ylabel('Power of main engine (modified)')

# Customize the scatter plot
ax.scatter(original_df['Power of main engine'], df['Power of main engine'], s=10, c='red', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Tonnage GT')
ax.set_xlabel('Tonnage GT (Original)')
ax.set_ylabel('Tonnage GT (modified)')

# Customize the scatter plot
ax.scatter(original_df['Tonnage GT'], df['Tonnage GT'], s=10, c='red', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()

df = df[df.Result == 'PRESUM']

variables = [#'Place of registration'#not necessary
            #'IRCS indicator', #delete
            #'Licence indicator', #delete
            #'VMS indicator', #delete
            #'ERS indicator', #delete
            #'AIS indicator', #delete
            'Vessel Type', # too important mantain, maybe supress few
            'Main fishing gear', # too important mantain
            #'Subsidiary fishing gear 1', # delete
            #'Subsidiary fishing gear 2', # delete
            #'Subsidiary fishing gear 3', # delete
            #'Subsidiary fishing gear 4', # delete
            #'Subsidiary fishing gear 5', # delete
            #'LOA', # noise addition
            #'LBP', # noise addition
            #'Tonnage GT', # noise addition
            'Other tonnage', # rounding nearest integer
            #'Power of main engine', # noise addition
            'Power of auxiliary engine', # rounding nearest tens
            'Hull material', # too important mantain
            #'Date of entry into service', #noise addition
            #'Segment', #delete
            #'Public aid', #delete
            #'Year of construction' #noise addition
            ]

grouped = CFR_match.groupby(variables).size().to_dict()
CFR_match['freq'] = CFR_match[variables].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

grouped = df.groupby(variables).size().to_dict()
df['freq'] = df[variables].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

freq = CFR_match[variables+['CFR','freq']]

freq_1 = CFR_match[variables+['CFR','freq']][CFR_match.freq==1]

freq_df_1 = df[variables+['CFR','freq']][df.freq==1]

freq_matched = freq.merge(freq_df_1, on=variables+['CFR'], how='inner')


print(freq_matched)

save_for_suppress = []

# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-1]):
    for var2 in variables[i+1:]:
        # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
        grouped = CFR_match.groupby([var1,var2]).size().to_dict()
        CFR_match['freq'] = CFR_match[[var1,var2]].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

        grouped = df.groupby([var1,var2]).size().to_dict()
        df['freq'] = df[[var1,var2]].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

        freq = CFR_match[[var1,var2]+['CFR','freq']]

        freq_1 = CFR_match[[var1,var2]+['CFR','freq']][CFR_match.freq==1]

        freq_df_1 = df[[var1,var2]+['CFR','freq']][df.freq==1]

        freq_matched = freq.merge(freq_df_1, on=[var1,var2]+['CFR'], how='inner')

        freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][[var1, var2, 'CFR','freq_x','freq_y']]
        # Check if the DataFrame is empty, and print a message if it is not
        if not freq_1_df.empty:
            print(f"Pairs of {var1} and {var2} with a frequency of 1:\n{freq_1_df}")
            save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-2]):
    for j, var2 in enumerate(variables[i+1:-1]):
        for var3 in variables[i+j+2:]:
            # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
            variables_sel = [var1,var2,var3]
            grouped = CFR_match.groupby(variables_sel).size().to_dict()
            CFR_match['freq'] = CFR_match[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

            grouped = df.groupby(variables_sel).size().to_dict()
            df['freq'] = df[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

            freq = CFR_match[variables_sel+['CFR','freq']]

            freq_1 = CFR_match[variables_sel+['CFR','freq']][CFR_match.freq==1]

            freq_df_1 = df[variables_sel+['CFR','freq']][df.freq==1]

            freq_matched = freq.merge(freq_df_1, on=variables_sel+['CFR'], how='inner')

            freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][variables_sel+['CFR','freq_x','freq_y']]
            # Check if the DataFrame is empty, and print a message if it is not
            if not freq_1_df.empty:
                print(f"Pairs of {variables_sel} with a frequency of 1:\n{freq_1_df}")
                save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)

# Iterate over all variables
for var in variables:
    # Create a new DataFrame that only contains rows with a frequency of 1 for the variable
    CFR_match['freq'] = CFR_match[var].map(CFR_match[var].value_counts())
    df['freq'] = df[var].map(df[var].value_counts())

    freq = CFR_match[[var, 'CFR', 'freq']]
    freq_1 = CFR_match[(CFR_match.freq == 1)][[var, 'CFR', 'freq']]
    freq_df_1 = df[df.freq == 1][[var, 'CFR', 'freq']]

    freq_matched = freq.merge(freq_df_1, on=[var, 'CFR'], how='inner')

    freq_1_df = freq_matched[(freq_matched['freq_x'] == 1) & (freq_matched['freq_y'] == 1)][[var, 'CFR', 'freq_x', 'freq_y']]

    # Check if the DataFrame is not empty and update 'freq' column in CFR_match
    if not freq_1_df.empty:
        CFR_match.loc[CFR_match['CFR'].isin(freq_1_df['CFR']), 'freq'] = 1370

        # Print the result
        print(f"Variable {var} with a frequency of 1:\n{freq_1_df}")
        save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-3]):
    for j, var2 in enumerate(variables[i+1:-2]):
        for k, var3 in enumerate(variables[i+j+2:-1]):
            for var4 in variables[k+i+j+3:]:
                # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
                variables_sel = [var1,var2,var3, var4]
                grouped = CFR_match.groupby(variables_sel).size().to_dict()
                CFR_match['freq'] = CFR_match[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

                grouped = df.groupby(variables_sel).size().to_dict()
                df['freq'] = df[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

                freq = CFR_match[variables_sel+['CFR','freq']]

                freq_1 = CFR_match[variables_sel+['CFR','freq']][CFR_match.freq==1]

                freq_df_1 = df[variables_sel+['CFR','freq']][df.freq==1]

                freq_matched = freq.merge(freq_df_1, on=variables_sel+['CFR'], how='inner')

                freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][variables_sel+['CFR','freq_x','freq_y']]
                # Check if the DataFrame is empty, and print a message if it is not
                if not freq_1_df.empty:
                    print(f"Pairs of {variables_sel} with a frequency of 1:\n{freq_1_df}")
                    save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


pd.DataFrame(save_for_suppress,columns=['CFR']).to_pickle(r'data\supress.pickle')

save_for_suppress = set(save_for_suppress)

variables = ['Other tonnage', 'Vessel Type', 'Vessel Type', 'Vessel Type', 'Power of auxiliary engine','Vessel Type']

index_supress = 0
for CFR in save_for_suppress:
    mask = df['CFR'] == CFR
    older_value = df.loc[mask, variables[index_supress]].values[0]
    
    # Check if the value is a string
    if isinstance(older_value, str):
        df.loc[mask, variables[index_supress]] = np.nan
    else:
        closest_values = df[variables[index_supress]].drop_duplicates().sort_values(key=lambda x: abs(x - older_value)).iloc[1:4]
        new_value = closest_values.mean()
        df.loc[mask, variables[index_supress]] = np.round(new_value,0)
        print(CFR)
        print(variables[index_supress])
        print(new_value)
        print(older_value)
    
    index_supress += 1


variables = [#'Place of registration'#not necessary
            #'IRCS indicator', #delete
            #'Licence indicator', #delete
            #'VMS indicator', #delete
            #'ERS indicator', #delete
            #'AIS indicator', #delete
            'Vessel Type',
            'Main fishing gear', # too important mantain
            #'Subsidiary fishing gear 1', # delete
            #'Subsidiary fishing gear 2', # delete
            #'Subsidiary fishing gear 3', # delete
            #'Subsidiary fishing gear 4', # delete
            #'Subsidiary fishing gear 5', # delete
            #'LOA', # noise addition
            #'LBP', # noise addition
            #'Tonnage GT', # noise addition
            'Other tonnage', # rounding nearest integer
            #'Power of main engine', # noise addition
            'Power of auxiliary engine', # rounding nearest tens
            'Hull material', # too important mantain
            #'Date of entry into service', #noise addition
            #'Segment', #delete
            #'Public aid', #delete
            #'Year of construction' #noise addition
            ]

grouped = CFR_match.groupby(variables).size().to_dict()
CFR_match['freq'] = CFR_match[variables].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

grouped = df.groupby(variables).size().to_dict()
df['freq'] = df[variables].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

freq = CFR_match[variables+['CFR','freq']]

freq_1 = CFR_match[variables+['CFR','freq']][CFR_match.freq==1]

freq_df_1 = df[variables+['CFR','freq']][df.freq==1]

freq_matched = freq.merge(freq_df_1, on=variables+['CFR'], how='inner')


print(freq_matched)

save_for_suppress = []

# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-1]):
    for var2 in variables[i+1:]:
        # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
        grouped = CFR_match.groupby([var1,var2]).size().to_dict()
        CFR_match['freq'] = CFR_match[[var1,var2]].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

        grouped = df.groupby([var1,var2]).size().to_dict()
        df['freq'] = df[[var1,var2]].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

        freq = CFR_match[[var1,var2]+['CFR','freq']]

        freq_1 = CFR_match[[var1,var2]+['CFR','freq']][CFR_match.freq==1]

        freq_df_1 = df[[var1,var2]+['CFR','freq']][df.freq==1]

        freq_matched = freq.merge(freq_df_1, on=[var1,var2]+['CFR'], how='inner')

        freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][[var1, var2, 'CFR','freq_x','freq_y']]
        # Check if the DataFrame is empty, and print a message if it is not
        if not freq_1_df.empty:
            print(f"Pairs of {var1} and {var2} with a frequency of 1:\n{freq_1_df}")
            save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-2]):
    for j, var2 in enumerate(variables[i+1:-1]):
        for var3 in variables[i+j+2:]:
            # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
            variables_sel = [var1,var2,var3]
            grouped = CFR_match.groupby(variables_sel).size().to_dict()
            CFR_match['freq'] = CFR_match[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

            grouped = df.groupby(variables_sel).size().to_dict()
            df['freq'] = df[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

            freq = CFR_match[variables_sel+['CFR','freq']]

            freq_1 = CFR_match[variables_sel+['CFR','freq']][CFR_match.freq==1]

            freq_df_1 = df[variables_sel+['CFR','freq']][df.freq==1]

            freq_matched = freq.merge(freq_df_1, on=variables_sel+['CFR'], how='inner')

            freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][variables_sel+['CFR','freq_x','freq_y']]
            # Check if the DataFrame is empty, and print a message if it is not
            if not freq_1_df.empty:
                print(f"Pairs of {variables_sel} with a frequency of 1:\n{freq_1_df}")
                save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)

# Iterate over all variables
for var in variables:
    # Create a new DataFrame that only contains rows with a frequency of 1 for the variable
    CFR_match['freq'] = CFR_match[var].map(CFR_match[var].value_counts())
    df['freq'] = df[var].map(df[var].value_counts())

    freq = CFR_match[[var, 'CFR', 'freq']]
    freq_1 = CFR_match[(CFR_match.freq == 1)][[var, 'CFR', 'freq']]
    freq_df_1 = df[df.freq == 1][[var, 'CFR', 'freq']]

    freq_matched = freq.merge(freq_df_1, on=[var, 'CFR'], how='inner')

    freq_1_df = freq_matched[(freq_matched['freq_x'] == 1) & (freq_matched['freq_y'] == 1)][[var, 'CFR', 'freq_x', 'freq_y']]

    # Check if the DataFrame is not empty and update 'freq' column in CFR_match
    if not freq_1_df.empty:
        CFR_match.loc[CFR_match['CFR'].isin(freq_1_df['CFR']), 'freq'] = 1370

        # Print the result
        print(f"Variable {var} with a frequency of 1:\n{freq_1_df}")
        save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


# Iterate over all pairs of two variables
for i, var1 in enumerate(variables[:-3]):
    for j, var2 in enumerate(variables[i+1:-2]):
        for k, var3 in enumerate(variables[i+j+2:-1]):
            for var4 in variables[k+i+j+3:]:
                # Create a new DataFrame that only contains rows with a frequency of 1 for both variables
                variables_sel = [var1,var2,var3, var4]
                grouped = CFR_match.groupby(variables_sel).size().to_dict()
                CFR_match['freq'] = CFR_match[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

                grouped = df.groupby(variables_sel).size().to_dict()
                df['freq'] = df[variables_sel].apply(lambda x: grouped.get(tuple(x), 0), axis=1)

                freq = CFR_match[variables_sel+['CFR','freq']]

                freq_1 = CFR_match[variables_sel+['CFR','freq']][CFR_match.freq==1]

                freq_df_1 = df[variables_sel+['CFR','freq']][df.freq==1]

                freq_matched = freq.merge(freq_df_1, on=variables_sel+['CFR'], how='inner')

                freq_1_df = freq_matched[(freq_matched['freq_x']==1) & (freq_matched['freq_y']==1)][variables_sel+['CFR','freq_x','freq_y']]
                # Check if the DataFrame is empty, and print a message if it is not
                if not freq_1_df.empty:
                    print(f"Pairs of {variables_sel} with a frequency of 1:\n{freq_1_df}")
                    save_for_suppress = save_for_suppress+list(freq_1_df.CFR.values)


pd.DataFrame(save_for_suppress,columns=['CFR']).to_pickle(r'data\supress2.pickle')

save_for_suppress = set(save_for_suppress)

print(save_for_suppress)

variables = ['Vessel Type', 'Other tonnage','Power of auxiliary engine','Vessel Type']

index_supress = 0
for CFR in save_for_suppress:
    mask = df['CFR'] == CFR
    older_value = df.loc[mask, variables[index_supress]].values[0]
    
    # Check if the value is a string
    if isinstance(older_value, str):
        df.loc[mask, variables[index_supress]] = np.nan
    else:
        closest_values = df[variables[index_supress]].drop_duplicates().sort_values(key=lambda x: abs(x - older_value)).iloc[1:4]
        new_value = closest_values.mean()
        df.loc[mask, variables[index_supress]] = np.round(new_value,0)
        print(CFR)
        print(variables[index_supress])
        print(new_value)
        print(older_value)
    
    index_supress += 1




