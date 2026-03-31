from fredapi import Fred
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

# ── Data Collection ────────────────────────────────────────────────
# Brent crude oil prices from FRED, annualized to yearly averages
fred = Fred(api_key='YOUR_API_KEY_HERE')
oil = fred.get_series('DCOILBRENTEU').resample('YE').mean()
oil = pd.DataFrame(oil)
oil.index = oil.index.year
oil.columns = ['Oil Price']

# Kuwait GDP per capita growth from World Bank
gdp_growth = pd.read_csv('data/API_NY.GDP.PCAP.KD.ZG.csv', skiprows=4)
kwt = gdp_growth[gdp_growth['Country Code'] == 'KWT']
kwt = kwt.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code', 'Unnamed: 70'])
kwt = pd.melt(kwt, id_vars=['Country Code'], var_name='Year', value_name='GDP Per Capita Growth')
kwt['Year'] = kwt['Year'].astype(int)

# ── Sample Period: 1993–2022 ───────────────────────────────────────
# 1992 excluded — post-Gulf War reconstruction produced an anomalous
# 51% GDP growth spike that obscures the oil-GDP relationship
kwt = kwt[(kwt['Year'] >= 1993) & (kwt['Year'] <= 2022)]
oil = oil[(oil.index >= 1993) & (oil.index <= 2022)]

# ── Merge & Clean ─────────────────────────────────────────────────
data = pd.merge(kwt, oil, left_on='Year', right_index=True)
data = data.set_index('Year').drop(columns=['Country Code'])

# ── VAR Model ─────────────────────────────────────────────────────
model = VAR(data)
results = model.fit(maxlags=1)
print(results.summary())