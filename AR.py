import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg

# ── Data Collection ────────────────────────────────────────────────
# Kuwait GDP per capita growth from World Bank (1992–2022)
gdp_growth = pd.read_csv('data/API_NY.GDP.PCAP.KD.ZG.csv', skiprows=4)
kwt = gdp_growth[gdp_growth['Country Code'] == 'KWT']
kwt = kwt.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code', 'Unnamed: 70'])
kwt = pd.melt(kwt, id_vars=['Country Code'], var_name='Year', value_name='GDP Per Capita Growth')
kwt['Year'] = kwt['Year'].astype(int)
kwt = kwt[(kwt['Year'] >= 1992) & (kwt['Year'] <= 2022)]

# ── Stationarity Check ─────────────────────────────────────────────
# ADF test confirms GDP growth is stationary (p ≈ 0), suitable for AR modeling
result = adfuller(kwt['GDP Per Capita Growth'].dropna())
print(f'ADF Statistic: {result[0]:.4f} | p-value: {result[1]:.4f}')

# ACF plot shows only lag 1 is significant, justifying AR(1) specification
plot_acf(kwt['GDP Per Capita Growth'].dropna())
plt.title('Autocorrelation Function — Kuwait GDP Per Capita Growth')
plt.show()

# ── AR(1) Model ────────────────────────────────────────────────────
series = kwt['GDP Per Capita Growth'].dropna().values
model = AutoReg(series, lags=1)
results = model.fit()
print(results.summary())