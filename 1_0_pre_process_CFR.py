import pandas as pd
import numpy as np

# Read CFR data from a pickle file

CFR_data = pd.read_pickle('data\CFR_data.pickle')

# Rename the column 'Country of importation/exportation' to 'Country_imp_exp'
CFR_data = CFR_data.rename(columns = {"Country of importation/exportation":"Country_imp_exp"})

# replace (DECONHECIDO) and (DESCONHECIDO) with (Unknown)
CFR_data = CFR_data.replace(['(DECONHECIDO)', '(DESCONHECIDO)'], 'Unknown')


# Save the modified DataFrame to a new pickle file
CFR_data.to_pickle('data\CFR_data.pickle')

# Get unique CFR values and sort them
CFR_unique = CFR_data.CFR.unique()
CFR_unique.sort()

# Group the DataFrame by 'Name of vessel' and perform aggregations
grouping = CFR_data.groupby(['Name of vessel']).agg({
    'CFR': lambda x: list(x.unique()),
    'Registration Number': lambda x: list(x.unique()),
    'Place of registration': lambda x: list(x.unique()),
}).reset_index()


# Create a DataFrame with unique vessel names and their corresponding information
Names_unique = pd.DataFrame({
    'Name' : grouping['Name of vessel'],
    'CFR_list' : grouping['CFR'],
    'Reg_Num_list': grouping['Registration Number'],
    'Place_code': grouping['Place of registration'],
    })

# Add a new column to the Names_unique DataFrame to store the number of unique CFRs per vessel
Names_unique['Num_CFR'] = [len(x) for x in Names_unique.CFR_list]


# Save the Names_unique DataFrame to a pickle file
Names_unique.to_pickle('data\grouped_CFR.pickle')
Names_unique.to_excel('data\grouped_CFR.xlsx')      



# Group the CFR_data DataFrame by the CFR column and create a list of unique places of registration
CFR_to_local_grouping = CFR_data.groupby(['CFR']).agg({
    'Place of registration': lambda x: list(x.unique()),
}).reset_index()

# Create a DataFrame that maps CFRs to their corresponding list of locals (place of registration)
CFR_to_local = pd.DataFrame({'CFR': CFR_to_local_grouping['CFR'],
                             'list_of_locals': CFR_to_local_grouping['Place of registration']
                             })

# Add a new column to the CFR_to_local DataFrame to store the number of unique places of registration per CFR
CFR_to_local['Num_locals'] = [len(x) for x in CFR_to_local['list_of_locals']]

# Add columns to store the last and previous licenses for each CFR
CFR_to_local['loc_old_new'] = [local[-2:] for local in CFR_to_local.list_of_locals]
CFR_to_local['Last_License'] = [x[-1] for x in CFR_to_local.loc_old_new]
CFR_to_local['Previous_License'] = [x[-2] if len(x)>1 else x[-1] for x in CFR_to_local.loc_old_new]

# Get the unique last licenses
loc_to_loc = CFR_to_local.Last_License.unique()

# Initialize an empty list to store the previous licenses corresponding to the last licenses
loc_to_loc2 = []

# Iterate through the unique last licenses and find the corresponding previous licenses
for loc in loc_to_loc:
    loc_to_loc2.append(CFR_to_local[CFR_to_local.Last_License==loc].Previous_License.unique()[0])

# Create a DataFrame to store the last and previous license pairs
loc_to_loc_pair = pd.DataFrame({'Last_License': loc_to_loc, 'Previous_License': loc_to_loc2})

# transform all to strings and indicate if it is portuguese or not
loc_to_loc_pair['Last_License'] = loc_to_loc_pair['Last_License'].astype(str)
loc_to_loc_pair['PT_indicator'] = [1 if license.startswith('PT') else 0 for license in loc_to_loc_pair.Last_License]


# Save the loc_to_loc_pair DataFrame to a pickle file
loc_to_loc_pair.to_pickle('data\loc_to_loc_pair.pickle')
loc_to_loc_pair.to_excel('data\loc_to_loc_pair.xlsx')

# Save the CFR_to_local DataFrame to a pickle file
CFR_to_local.to_pickle('data\CFR_to_local.pickle')
CFR_to_local.to_excel('data\CFR_to_local.xlsx')