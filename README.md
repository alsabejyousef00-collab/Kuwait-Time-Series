# Kuwait GDP Growth — Time Series Analysis

## Research Question
Does Kuwait's past GDP per capita growth predict its current growth, and does oil price play a measurable role in short-term economic performance?

## Methodology
This project applies three time series models to Kuwait's GDP per capita growth rate from 1992 to 2022, using World Bank and FRED data.

**AR(1):** An autoregressive model using one lag of GDP growth as a predictor. Every 1 percentage point increase in Kuwait's GDP growth in a given year is associated with approximately 0.5 percentage point higher growth the following year.

**ARIMA(1,0,0):** Extends the AR model to account for past forecast errors. The moving average term was statistically insignificant, confirming the simpler AR(1) specification. The autoregressive coefficient remained stable at approximately 0.5.

**VAR(1):** A vector autoregression modeling Kuwait GDP growth and Brent crude oil prices jointly. Oil prices were statistically insignificant as a predictor of Kuwait's short-term GDP growth. Kuwait's own past growth remained the dominant predictor.

## Key Finding
Oil price surges do not translate into immediate domestic GDP growth in Kuwait. Kuwait channels oil revenue into the Kuwait Investment Authority and deploys it internationally rather than domestically, creating a structural disconnect between oil wealth and local economic activity. This is not a failure of the model but a genuine institutional finding.

## Data Sources
World Bank: GDP per capita growth (NY.GDP.PCAP.KD.ZG)
FRED: Brent crude oil prices (DCOILBRENTEU)

## Notes
Add your FRED API key at the designated line in VAR.py. 