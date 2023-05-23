import pandas as pd
import numpy as np

# load fiscrep
fiscrep = pd.read_pickle('data\merged_fiscrep.pickle')



# load CFR_data european
CFR_data = pd.read_pickle('data\CFR_data.pickle')

# Remove duplicates
CFR_data.drop_duplicates(subset='CFR', keep='last', inplace=True)

# A list of variables to select
CFR_data = CFR_data[['CFR',
                    #'Event', 
                    #'Event Start Date', 
                    #'Event End Date', 
                    'Registration Number', 
                    #'External marking', 
                    #'Name of vessel', 
                    'Place of registration', 
                    #'IRCS', 
                    #'IRCS indicator', 
                    #'Licence indicator', 
                    #'VMS indicator', 
                    #'ERS indicator', 
                    #'AIS indicator', 
                    #'MMSI', 
                    'Vessel Type', 
                    'Main fishing gear', 
                    'LOA', 
                    'LBP', 
                    'Tonnage GT', 
                    'Other tonnage', 
                    #'GTs', 
                    'Power of main engine', 
                    'Power of auxiliary engine', 
                    'Hull material', 
                    #'Country_imp_exp', 
                    'Year of construction',
                    'Date of entry into service',
                    #'Active'
                    ]]

# Rename to merge in CFR_data
CFR_data.rename(columns = {'CFR': 'matched_CFR'},inplace=True)

# Use merge to join the two DataFrames on the 'CFR' column
merged_df = fiscrep.merge(CFR_data, on='matched_CFR', how='left')

# Some variable change
variable_change = {'Registration Number_x': 'Reg_Num',     
                    'Vessel_Type':'Vessel_Type_x', 
                    'Vessel Type':'Vessel_Type_y', 
}
merged_df.rename(columns = variable_change, inplace=True)

# check if all checkout (not printing is good)
for index, fiscalization in merged_df.iterrows():
    if fiscalization.CFR != 'Unknown':
        if fiscalization.CFR != fiscalization.matched_CFR:
            if fiscalization.matched_CFR.startswith('No_CFR'):
                print(fiscalization[['CFR','matched_CFR']])

# Create a column where we save the codes of the Reg_Number
merged_df['local'] = 'Foreign'

for index, fiscalization in merged_df.iterrows():
    if isinstance(fiscalization.Reg_Num,str):
        reg = fiscalization.Reg_Num.split('-')
        if len(reg) == 3:
            merged_df.local.iloc[index] = reg[0]
        else:
            print(fiscalization.Reg_Num)


# load the codes associate to the locals, previous and new
loc_to_loc = pd.read_pickle('data\loc_to_loc_pair.pickle')

local_convert = pd.DataFrame({'old': [], 'new': []})

# get the unique values of local codes, PT only
for index, local in loc_to_loc.iterrows():
    if local.Last_License.startswith('PT'):
        if not local.Previous_License.startswith('PT'):
            local_convert = local_convert.append({'old': local.Previous_License, 'new': local.Last_License}, ignore_index=True)

# merge to have the local where vessels where registered
merged_df['real_local'] = merged_df['local'].where(
    merged_df['local'].isin(['Foreign', 'PT']), 
    merged_df['local'].map(local_convert.set_index('old')['new'])
).fillna(merged_df['local'])

# Get rid of the foreign values and put NaN
merged_df['real_local'] = merged_df['real_local'].replace('Foreign', np.nan)
merged_df['local'] = merged_df['local'].replace('Foreign', np.nan)

# add the description of the LOCODES
local_description = pd.read_csv('data/table_locations.csv')[['LOCODE_T','NameWoDiacritics']]
merged_df = merged_df.merge(local_description, left_on='real_local', right_on='LOCODE_T', how='left')
merged_df = merged_df.drop(columns='LOCODE_T')
merged_df.rename(columns = {'NameWoDiacritics':'Local_Name'}, inplace=True)


#Separate the infractions by commas and save it in arrays
merged_df["Infrac_a"] = merged_df.Infraction.apply(lambda x: x.split())

# Create a set to hold all unique infractions in the DataFrame
all_infractions = set()

# Iterate over the Infrac column to get all unique infractions
for infrac_list in merged_df.Infrac_a:
    all_infractions.update(infrac_list)

# Define a dictionary that maps each infraction to its numeric value
infraction_values = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                     'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
                     'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14}

# Sort the keys of the dictionary based on their values
sorted_infractions = sorted(infraction_values, key=infraction_values.get)


# Create indicator columns for each infraction in the sorted order
for infraction in sorted_infractions:
    merged_df[infraction] = merged_df.Infrac_a.apply(lambda x: infraction in x).astype(int)


# more variables created
merged_df['number_infracs'] = [len(infrac_a) for infrac_a in merged_df.Infrac_a]
merged_df['Year'] = [data.year for data in merged_df.GDH]
merged_df['Month'] = [data.month for data in merged_df.GDH]
merged_df['Day'] = [data.day for data in merged_df.GDH]
merged_df['Hour'] = [round(data.hour+data.minute/60,2) for data in merged_df.GDH]
merged_df['Period'] = [int(np.floor(1+hour/6)) for hour in merged_df.Hour]

# create the column related to NUTSII
nuts_codes = pd.read_excel(r'data\para_nuts_complete.xlsx')
NutsII_Code = []
for i, vessel in merged_df.iterrows():
    local = merged_df.Local_Name[i]
    if local == local:
        code = nuts_codes[nuts_codes.Local == local].Code.values[0]
        NutsII_Code.append(code)
    else:
        NutsII_Code.append(local)
merged_df['NUTSII_Code'] = NutsII_Code

# save merged data
merged_df.to_csv('data/final_fiscrep.csv')
merged_df.to_pickle('data/final_fiscrep.pickle')


selected_variables = ['Name', 'Reg_Num', 'Latitude', 'Longitude', 'GDH',
       'Unit', 'Vessel_Type_x', 'Sub_Type', 'Art', 'Result',
       'Infraction', 
       'matched_CFR',
       'Registration Number_y', 'Place of registration',
       #'Licence indicator', 'VMS indicator', 'ERS indicator', 'AIS indicator',
       'Vessel_Type_y', 'Main fishing gear',
       #'Subsidiary fishing gear 1', 'Subsidiary fishing gear 2',
       #'Subsidiary fishing gear 3', 'Subsidiary fishing gear 4',
       #'Subsidiary fishing gear 5',
       'LOA', 'LBP', 'Tonnage GT',
       'Other tonnage', 'Power of main engine',
       'Power of auxiliary engine', 'Hull material',
       'Date of entry into service', 'Year of construction', 'local',
       'real_local', 'Local_Name', 'Infrac_a', 'I', 'II', 'III', 'IV', 'V',
       'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
       'number_infracs', 'Year', 'Month', 'Day', 'Hour', 'Period','NUTSII_Code']

merged_df_selected = merged_df[selected_variables]

merged_df_selected.to_csv('data/final_fiscrep_selected.csv')
merged_df_selected.to_pickle('data/final_fiscrep_selected.pickle')