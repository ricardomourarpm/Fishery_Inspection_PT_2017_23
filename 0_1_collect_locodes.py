# Import the pandas library to work with dataframes
import pandas as pd

# URL of the webpage with the table to scrape
url = 'https://service.unece.org/trade/locode/pt.htm'

# Extract all tables from the webpage using pandas read_html function
tables = pd.read_html(url)

# Get the third table (index 2) from the tables list and remove the first row
df = tables[2].iloc[1:]

# Get the column names from the first row of the same table and create a list
names_html = list(tables[2].iloc[0])

# Create a new dataframe with the column names obtained
df = pd.DataFrame(df.values, columns=names_html)

# Create a new column 'LOCODE_T' with cleaned data from the 'LOCODE' column
df['LOCODE_T'] = [code.replace(" ","") for code in df.LOCODE]

# Save the final dataframe as a CSV file, without the index column

df.to_csv(r'data\table_locations.csv', index=False)