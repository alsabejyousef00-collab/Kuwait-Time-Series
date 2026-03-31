from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR 

# Import Data

fred = Fred(api_key='49eaf9f6a620ac8cb4d772261ecfc638')

oil = fred.get_series('DCOILBRENTEU')
# Annualize oil
oil_annual = oil.resample('YE').mean()

#Get all data into dataframe

gdp_growth = pd.read_csv('/Users/yousefalsabej/Documents/Work/Python/Independent Research Projects/Excercise Consolidation Pt.1/GDP Per Capita % GROWTH/API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_94752.csv',skiprows=4)
oil = pd.DataFrame(oil_annual)

#Filter gdp_growth for kwt

kwt = gdp_growth[gdp_growth['Country Code'] == 'KWT']

# Cleanse data and get ready for VAR test.

kwt = kwt.drop(columns=['Country Name','Indicator Name', 'Indicator Code', 'Unnamed: 70'])

oil.index =oil.index.year
oil.columns = ['Oil Price']

# Melt KWT Data 
kwt = pd.melt(kwt, id_vars=['Country Code'],var_name='Year',value_name='GDP Per Capita Growth')

#Convert Years into integers

kwt['Year'] = kwt['Year'].astype(int)

#Filter both data frames for between 1992-2022
kwt = kwt[(kwt['Year'] >= 2010)& (kwt['Year'] <= 2022)]
oil = oil[(oil.index >=2010)& (oil.index <=2022)]


# Merge the dataframes

data = pd.merge(kwt, oil, left_on='Year', right_index=True)

# Final cleanse

data = data.set_index('Year')

data = data.drop(columns=['Country Code'])

# Let's run the VAR model 

model = VAR(data)
results = model.fit(maxlags=1)
print(results.summary())
