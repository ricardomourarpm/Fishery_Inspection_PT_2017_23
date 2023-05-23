import pandas as pd

# convert excel to pickle to read faster

CFR_data = pd.read_excel('data\CFR_data.xlsx')

CFR_data_Esp = pd.read_excel('data\CFR_data_ESP.xlsx')

# concat spanish vessels
CFR_data = pd.concat((CFR_data,CFR_data_Esp))

CFR_data.to_pickle('data\CFR_data.pickle')