import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

# Import Data

gdp_growth = pd.read_csv('/Users/yousefalsabej/Documents/Work/Python/Independent Research Projects/Excercise Consolidation Pt.1/GDP Per Capita % GROWTH/API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_94752.csv',skiprows=4)

#Filter only for Kuwait

kwt = gdp_growth[gdp_growth['Country Code']=='KWT']


# Clean up the data a little more
kwt = kwt.drop(columns=['Country Name','Indicator Name', 'Indicator Code', 'Unnamed: 70'])


# Melt the data
kwt = pd.melt(kwt, id_vars=['Country Code'], var_name='Year',value_name='GDP Per Capita Growth')

# Convert Year to Integer

kwt['Year'] = kwt['Year'].astype(int)


#Filter for between 1992-2022, or Kuwait's modern Economy Post-Gulf War

kwt = kwt[(kwt['Year'] >= 1992) & (kwt['Year'] <= 2022)]


# Let's plot Kuwait's gdp per capita Growth  over the period of 1992-2022

# plt.plot(kwt['Year'], kwt['GDP Per Capita Growth'])
# plt.xticks(rotation=45, fontsize=8)
# plt.show()


# Confirm that data is stationary so that we can build Time Effect / AR Model.

# result = adfuller(kwt['GDP Per Capita Growth'].dropna())
# print('ADF Statistic:', result[0])
# print('p-value',result[1])

# Quick ACF check to check ow many lags are significant.

# plot_acf(kwt['GDP Per Capita Growth'].dropna())
# plt.show()

# ARIMA Test
series = kwt['GDP Per Capita Growth'].dropna().values
model = ARIMA(series, order=(1,0,0))
results = model.fit()
print(results.summary())