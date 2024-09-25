import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load the dataset
file_path = '../data/time_series_data.csv'  # Path to your CSV file
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data = data.asfreq('D')
# Visualize the dataset
# plt.figure(figsize=(10, 5))
# plt.plot(data.index, data['Value'], label='Actual Values')
# plt.title('Time Series Data')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()

# Fit an ARIMA model
model = ARIMA(data['Value'], order=(4, 1, 0))
model_fit = model.fit()

print(data['Value'][-1])

# Make predictions for the next 10 days
forecast_steps = 10
forecast = model_fit.forecast(steps = forecast_steps)
forecast = [data['Value'][-1]] + list(forecast)

# Create a DataFrame for the forecasted values
forecast_index = [data.index[-1] + timedelta(days=i) for i in range(0, forecast_steps + 1)]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecasted Value'])
print(forecast_df)

# Visualize the actual vs. forecasted values
plt.figure(figsize=(15, 5))
plt.plot(data.index, data['Value'], label='Actual Values')
plt.plot(forecast_df.index, forecast_df['Forecasted Value'], label='Forecasted Values', color='red', linestyle='--')
plt.title('Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()
