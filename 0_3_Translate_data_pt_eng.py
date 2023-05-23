import pandas as pd

# read data from Excel file
fiscrep = pd.read_excel(r'data\FISCREP 2015-2023.xls', sheet_name='RELATOS')

# clean the headers of anyspace after the last letter
fiscrep = fiscrep.rename(columns=lambda x: x.rstrip())

# Remove all spaces in the entire dataframe
fiscrep = fiscrep.applymap(lambda x: x.strip() if isinstance(x, str) else x)


#get the original columns labels
pt_labels = fiscrep.columns

# translation dictionary
original_list = ['Nº', 'Nome', 'CFR', 'nº reg', 'Latitude', 'Longitude', 'GDH',
                 'Unidade', 'FIS', 'Tip Emb', 'Sub Tip', 'Arte', 'Result', 'Infrac']

translated_list = ['Number', 'Name', 'CFR', 'Reg_Number', 'Latitude', 'Longitude', 'GDH',
                   'Unit', 'FIS', 'Vessel_Type', 'Sub_Type', 'Art', 'Result', 'Infraction']

# Create a dictionary by zipping the two lists together
dictionary = dict(zip(original_list, translated_list))

# Rename columns using the translation dictionary
fiscrep = fiscrep.rename(columns=dictionary)

#CFR translation
translation_dict = {'DESCONHECIDO':'Unknown'}
fiscrep['CFR'] = fiscrep['CFR'].replace(translation_dict)

#Vessel_Type translation
fiscrep['Vessel_Type'].unique()

# Translation dictionary for 'Vessel_Type'
translation_dict = {
    'Pesca comercial': 'Commercial fishing',
    'Recreio': 'Recreational',
    'Maritimo turisticas': 'Touristic maritime',
    'Artes caladas': 'Fixed gear',
    'Outras': 'Others',
    'Desconhecido': 'Unknown'
}
fiscrep['Vessel_Type'] = fiscrep['Vessel_Type'].replace(translation_dict)

#Sub_Type tranlation
# Translation dictionary for 'Sub_Type'
translation_dict = {
    'ARRASTÃO': 'Trawl', ##
    'ARMADILHAS': 'Traps', ##
    'Desconhecido': 'Unknown', ##
    'POLIVALENTE': 'Multipurpose', ##
    'EMALHAR/TRESMALHO': 'Gillnet/trammel net', ##
    'PALANGREIRO': 'Longline', ##
    'GANCHORRA': 'Towed Dredge', ##
    'NAVIO DE PESCA À LINHA': 'Line fishing vessel', ##
    'SALTO E VARA': 'Pole and line', ##
    'OUTRAS': 'Other', ##
    'CERCADOR': 'Seine net', ##
    'Pesca turistica': 'Touristic fishing', ##
    'Covos pequeno': 'Small crab pot', ##
    'Covos m pequeno': 'Small lobster pot', ##
    'Aluguer com tripulação': 'Crewed rental', ##
    'Serviços de reboque recreativo': 'Recreational towing services', ##
    'Palangre grande': 'Big longline', ##
    'Tresmalho mto pequeno': 'Very small trammel net', ##
    'NAVIO DE APOIO': 'Support vessel', ##
    'Palangre mto pequeno': 'Very small longline', ##
    'APANHA DE ALGAS': 'Seaweed harvesting', ##
    'Alcatruzes pequen': 'Small traps net', 
    'Passeios com programa': 'Programmed rides', ##
    'Aluguer sem tripulação': 'Uncrewed rental', ##
    'Alcatruzes grande': 'Big traps net', ##
    'Alcatruzes m peq': 'Medium-small traps net', ##
    'Murejona m pequeno': 'Small fish pots', ##
    'Todos': 'All', ##
    'Emalho muito pequeno': 'Very small gillnet', ##
    'Boscas mto pequeno': 'Very small pot', ##
    'Alcatruzes mg': 'Big traps net', ##
    'Emalho pequeno': 'Small gillnet', ##
    'NAVIO FÁBRICA': 'Factory vessel', ##
    'NÃO IDENTIFICADO': 'Unidentified', ##
    'Covos grande': 'Big crab trap', ##
    'Covos mto grande': 'Very big crab trap', ##
    'Emalho muito grande': 'Very big gillnet', ##
    'Águas abrigadas': 'Sheltered waters' ##
}
fiscrep['Sub_Type'] = fiscrep['Sub_Type'].replace(translation_dict)

#Art translation
# Translation dictionary for 'Art'
translation_dict = {
    'Arrasto': 'Trawl', ##
    'Armadil': 'Traps', ##
    'Linhas': 'Lines', ##
    'Emalhar': 'Gillnetting', ##
    'Tresmal': 'Trammel nets', ##
    'TODAS': 'All', ##
    'Palangr': 'Longline', ##
    'Linha s': 'Handline', ##
    'Ganchor': 'Towed Dredge', ##
    'Nassa,': 'Pots', ##
    'Com ret': 'Purse seine', ##
    'Alcatru': 'Traps bucket',
    'Mista d': 'Mixed gears', ##
    'Linha d': 'Dropline', 
    'Armação': 'Pound nets', ##
    'Sacada': 'Lift nets', ##
    'Cerco e': 'Scotish seines', ##
    'Cerco d': 'Danish seines', ##
    'Arpão': 'Harpoon', ##
    'Draga d': 'Hand Dredges', ##
    'Envolve': 'Seines',
    'Sem ret': 'Lampara nets',##
    'Artes d': 'Gear nei', ##
    'Redes d': 'Gillnetting', ##
}
fiscrep['Art'] = fiscrep['Art'].replace(translation_dict)

fiscrep.to_pickle(r'data\fiscrep_eng.pickle')
