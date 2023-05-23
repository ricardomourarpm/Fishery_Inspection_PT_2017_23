import pandas as pd
import numpy as np

df = pd.read_pickle(r'data\final_fiscrep_deleted.pickle')

o_df = df.copy()

# Create a new DataFrame for unique values of CFR
matched_CFR = pd.DataFrame(df.matched_CFR.unique(), columns=['CFR'])

df.rename(columns = {'matched_CFR':'CFR', 'Vessel_Type_y':'Vessel_Type'}, inplace = True)

df['Date of entry into service'] = df['Date of entry into service'].apply(lambda x: int(x.year) if not pd.isnull(x) else np.nan)

df['Year of construction'] = df['Year of construction'].apply(lambda x: int(x.year) if not pd.isnull(x) else np.nan)

# And more deletion after k-identifier analysis

df = df.drop(columns=['Place of registration',#not necessary
            #'IRCS indicator', #deleted
            #'Licence indicator', #deleted
            #'VMS indicator', #deleted
            #'ERS indicator', #deleted
            #'AIS indicator', #deleted
            #'Subsidiary fishing gear 1', # deleted
            #'Subsidiary fishing gear 2', # deleted
            #'Subsidiary fishing gear 3', # deleted
            #'Subsidiary fishing gear 4', # deleted
            #'Subsidiary fishing gear 5', # deleted
            #'Segment', #deleted
            #'Public aid', #deleted
            'Hour', #delete to only stay Period
            ])



seed_r = 100

np.random.seed(seed_r)
alpha = np.random.uniform(0.2,0.4)

sigma_date = np.std([x for x in df['Date of entry into service'] if not pd.isnull(x)])

sigma_year = np.std([x for x in df['Year of construction'] if not pd.isnull(x)])
np.random.seed(seed_r)
df['Date of entry into service'] = df['Date of entry into service'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_date),0) if not pd.isnull(x) else np.nan)
np.random.seed(seed_r)
df['Year of construction'] = df['Year of construction'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_year),0) if not pd.isnull(x) else np.nan)

sigma_LOA = df['LOA'].std()

sigma_LBP = df['LBP'].std()
np.random.seed(seed_r)
df['LOA']=df['LOA'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_LOA),1) if not pd.isnull(x) else np.nan)
np.random.seed(seed_r)
df['LBP']=df['LBP'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_LBP),1) if not pd.isnull(x) else np.nan)

df['LOA'] = [loa if loa>0 else o_df.LOA.min() if loa==0 else np.abs(loa) for loa in df['LOA']]

df['LBP'] = [lbp if lbp>0 else o_df.LBP.min() if lbp==0 else np.abs(lbp) for lbp in df['LBP']]

df['Power of auxiliary engine'] = (df['Power of auxiliary engine']/10).round()*10

df['Other tonnage'] = df['Other tonnage'].round()

sigma_power = df['Power of main engine'].std()
np.random.seed(seed_r)
df['Power of main engine'] = df['Power of main engine'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_power),1) if not pd.isnull(x) else np.nan)

df['Power of main engine'] = [power if power>0 else o_df['Power of main engine'].min() if power==0 else np.abs(power) for power in df['Power of main engine']]

sigma_ton = df['Tonnage GT'].std()
np.random.seed(seed_r)
df['Tonnage GT'] = df['Tonnage GT'].apply(lambda x: np.round(np.random.normal(loc=x, scale=alpha*sigma_ton),1) if not pd.isnull(x) else np.nan)

df['Tonnage GT'] = [ton if ton>0 else o_df['Tonnage GT'].min() if ton==0 else np.abs(ton) for ton in df['Tonnage GT']]

# Plot difference

import matplotlib.pyplot as plt

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Date of Entry into Service')
ax.set_xlabel('Date of entry into service (Original)')
ax.set_ylabel('Date of Entry into Service (modified)')

# Customize the scatter plot
ax.scatter(df['Date of entry into service'], o_df['Date of entry into service'], s=10, c='green', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()


#Plot difference

diff_year = o_df['Year of construction']-pd.to_datetime(df['Year of construction'], format='%Y')

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Year of Construction')
ax.set_xlabel('Year of Construction (Original)')
ax.set_ylabel('Year of Construction (modified)')

# Customize the scatter plot
ax.scatter(o_df['Year of construction'], df['Year of construction'], s=10, c='green', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()


# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of LOA')
ax.set_xlabel('LOA (Original)')
ax.set_ylabel('LOA (modified)')

# Customize the scatter plot
ax.scatter(o_df['LOA'], df['LOA'], s=10, c='blue', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of LBP')
ax.set_xlabel('LBP (Original)')
ax.set_ylabel('LBP (modified)')

# Customize the scatter plot
ax.scatter(o_df['LBP'], df['LBP'], s=10, c='blue', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()







# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Power of main engine')
ax.set_xlabel('Power of main engine (Original)')
ax.set_ylabel('Power of main engine (modified)')

# Customize the scatter plot
ax.scatter(o_df['Power of main engine'], df['Power of main engine'], s=10, c='red', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the plot title and axis labels
ax.set_title('Scatter plot of Tonnage GT')
ax.set_xlabel('Tonnage GT (Original)')
ax.set_ylabel('Tonnage GT (modified)')

# Customize the scatter plot
ax.scatter(o_df['Tonnage GT'], df['Tonnage GT'], s=10, c='red', alpha=0.5, edgecolors='none')

# Show the plot
plt.show()









save_for_suppress = pd.read_pickle(r'data\supress.pickle')

save_for_suppress = set(save_for_suppress.CFR)

variables = ['Other tonnage', 'Vessel_Type', 'Vessel_Type', 'Vessel_Type', 'Power of auxiliary engine','Vessel_Type']

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

save_for_suppress2 = pd.read_pickle(r'data\supress2.pickle')

save_for_suppress = set(save_for_suppress2.CFR)

variables = ['Power of auxiliary engine', 'Other tonnage']

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



# Create a list to store the assigned codes
CFR_code = []

# Initialize the code number
number = 1

# Assign a code to each unique value of CFR leaving the non-CFR unchanged
for CFR in matched_CFR.CFR:
    if CFR.startswith('NOCFR'):
        CFR_code.append(CFR)
    else:
        CFR_code.append('CFR_'+str(number))
        number += 1

# Add the assigned codes to the DataFrame
matched_CFR['code'] = CFR_code

# Replace the original values with the assigned codes in the original DataFrame
df['CFR'] = df['CFR'].map(dict(zip(matched_CFR['CFR'], matched_CFR['code'])))

# Create a new DataFrame for unique values of Unit
unit_unique = pd.DataFrame(df.Unit.unique(), columns=['Unit'])

# Create a list to store the assigned codes
unit_code = []

# Initialize the code number
number = 1

# Assign a code to each unique value of Unit
for unit in unit_unique['Unit']:
    unit_code.append('Unit_'+str(number))
    number += 1

# Add the assigned codes to the DataFrame
unit_unique['code'] = unit_code

# Replace the original values with the assigned codes in the original DataFrame
df['Unit'] = df['Unit'].map(unit_unique.set_index('Unit')['code']).fillna(df['Unit'])




import math
import random
from geopy import distance
import re

# Function to convert coordinates from degrees, minutes, and seconds format to decimal degrees
def dms_to_decimal(degrees, minutes, seconds):
    return degrees + minutes/60 + seconds/3600

# Function to convert decimal degrees to degrees, minutes, and seconds format
def decimal_to_dms(decimal_degrees):
    degrees = int(decimal_degrees)
    decimal_minutes = (decimal_degrees - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return f"{degrees}º{minutes}´{seconds:.2f}"

# Function to convert decimal degrees to degrees, minutes, and seconds format with direction
def decimal_to_dms_direction(decimal_degrees, direction):
    degrees = int(decimal_degrees)
    decimal_minutes = (decimal_degrees - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return f"{degrees}º{minutes}´{seconds:.2f}{direction}"

# Random displacement distance in miles
displacement_miles = 0.5

# Iterate over the Latitude and Longitude columns in the DataFrame
for index, row in df.iterrows():
    latitude_str = row['Latitude']
    longitude_str = row['Longitude']
    
    # Extract degrees, minutes, and seconds from the string using regular expressions
    latitude_parts = re.findall(r'\d+\.\d+|\d+', latitude_str)
    longitude_parts = re.findall(r'\d+\.\d+|\d+', longitude_str)
    
    # Convert degrees, minutes, and seconds to decimal degrees
    latitude = dms_to_decimal(float(latitude_parts[0]), float(latitude_parts[1]), float(latitude_parts[2]))
    longitude = dms_to_decimal(float(longitude_parts[0]), float(longitude_parts[1]), float(longitude_parts[2]))
    
    # Generate random displacement in latitude and longitude
    random_latitude_displacement = random.uniform(-displacement_miles, displacement_miles)
    random_longitude_displacement = random.uniform(-displacement_miles, displacement_miles)
    
    # Perform the displacement by converting the distance to kilometers
    new_latitude = latitude + (random_latitude_displacement / 69)
    new_longitude = longitude + (random_longitude_displacement / (69 * abs(math.cos(math.radians(latitude)))))
    
    # Convert the new latitude and longitude back to degrees, minutes, and seconds format with direction
    new_latitude_str = decimal_to_dms_direction(new_latitude, latitude_str[-1])
    new_longitude_str = decimal_to_dms_direction(new_longitude, longitude_str[-1])
    
    # Update the DataFrame with the new coordinates
    df.at[index, 'Latitude'] = new_latitude_str
    df.at[index, 'Longitude'] = new_longitude_str


import matplotlib.pyplot as plt
import geopandas as gpd

# Calculate the number of records to select for plotting (10% of the total records)
num_records = int(len(o_df) * 0.1)

# Randomly select 10% of the records from o_df and df
random_indices = random.sample(range(len(o_df)), num_records)
o_df_sample = o_df.iloc[random_indices]
df_sample = df.iloc[random_indices]

def convert_coordinates(coord):
    parts = re.split('[º´″]', coord[:-1])
    degrees = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    direction = coord[-1]

    decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal_degrees *= -1

    return decimal_degrees

# Extract latitude and longitude values from o_df and df
original_latitudes = o_df_sample['Latitude'].apply(convert_coordinates)
original_longitudes = o_df_sample['Longitude'].apply(convert_coordinates)
displaced_latitudes = df_sample['Latitude'].apply(convert_coordinates)
displaced_longitudes = df_sample['Longitude'].apply(convert_coordinates)

# Read the shapefile
shapefile_path = 'shapefiles\concelhos.shp'
data = gpd.read_file(shapefile_path)

# Plot the shapefile
data.plot()

# Plot the original and displaced coordinates
plt.scatter(original_longitudes, original_latitudes, color='blue', label='Original', s=10, alpha=0.5)
plt.scatter(displaced_longitudes, displaced_latitudes, color='red', label='Displaced', s=10, alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Original and Displaced Coordinates')
plt.legend()
plt.grid(True)

plt.show()

# some other preprocessing detected after
df.drop(columns=['Vessel_Type_x', 'Infrac_a','local'], inplace=True)

df['Result'].replace('TODOS', 'LEGAL', inplace=True)

df.rename(columns={'Art':'Gear'}, inplace=True)

#save fiscrep
df.to_pickle(r'data\final_fiscrep_anonimized.pickle')
df.to_csv(r'data\final_fiscrep_anonimized.csv')

o_df.to_pickle(r'data\final_fiscrep_original.pickle')
o_df.to_csv(r'data\final_fiscrep_original.csv')

df