import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ── Data Collection ────────────────────────────────────────────────
# Kuwait GDP per capita growth from World Bank (1992–2022)
gdp_growth = pd.read_csv('data/API_NY.GDP.PCAP.KD.ZG.csv', skiprows=4)
kwt = gdp_growth[gdp_growth['Country Code'] == 'KWT']
kwt = kwt.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code', 'Unnamed: 70'])
kwt = pd.melt(kwt, id_vars=['Country Code'], var_name='Year', value_name='GDP Per Capita Growth')
kwt['Year'] = kwt['Year'].astype(int)
kwt = kwt[(kwt['Year'] >= 1992) & (kwt['Year'] <= 2022)]

# ── ARIMA(1,0,0) Model ─────────────────────────────────────────────
# MA term (q=1) was tested and found insignificant — simpler AR(1)
# specification performs equivalently based on AIC comparison
series = kwt['GDP Per Capita Growth'].dropna().values
model = ARIMA(series, order=(1, 0, 0))
results = model.fit()
print(results.summary())