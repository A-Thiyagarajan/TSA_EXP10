### Name: Thiyagarajan A
### Reg.no: 212222240110
### Date: 
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement and enhance the SARIMA model using Python for accurately predicting the Google stock opening prices.
### ALGORITHM:
1. Import libraries, load the dataset, set the date column as datetime, and visualize the time series.
2. Use the Augmented Dickey-Fuller test to confirm stationarity and apply seasonal differencing if necessary.
3. Plot ACF and PACF on differenced data to identify potential SARIMA parameters.
4. Use auto_arima to find optimal (p, d, q) and seasonal (P, D, Q, S) parameters.
5. Divide data into training (80%) and testing (20%) sets.
6. Train the SARIMA model on the training data and generate predictions on the test set.
7. Calculate RMSE for accuracy and plot predictions against actual data to assess model performance.
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pmdarima as pm

# Load the Google stock price dataset
data = pd.read_csv('Google_Stock_Price_Train.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(data['Open'], color='blue')
plt.xlabel('Date')
plt.ylabel('Opening Stock Price')
plt.title('Google Stock Opening Price Time Series')
plt.show()

# Check stationarity of 'Open' prices
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Open'])

# Seasonal Differencing for Potentially Better Stationarity
data['Open_diff'] = data['Open'] - data['Open'].shift(12)
data.dropna(inplace=True)

# Plotting ACF and PACF
plot_acf(data['Open_diff'], lags=40)
plt.show()

plot_pacf(data['Open_diff'], lags=40)
plt.show()

# Step 1: Auto ARIMA for Best Parameters
auto_arima_model = pm.auto_arima(
    data['Open'],
    start_p=1, start_q=1, max_p=3, max_q=3,
    seasonal=True, m=12, start_P=1, start_Q=1, max_P=3, max_Q=3,
    d=1, D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(auto_arima_model.summary())

# Step 2: Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['Open'][:train_size], data['Open'][train_size:]

# Use the parameters suggested by Auto ARIMA
order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order

# Step 3: Fit the SARIMA model with the chosen parameters
sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit(disp=False)

# Step 4: Forecasting
predictions = sarima_result.predict(start=len(train), end=len(data)-1, dynamic=False)

# Step 5: Evaluate the model with RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Step 6: Plot Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Opening Stock Price')
plt.title('SARIMA Model Predictions for Google Stock Price')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/e2322c8b-5aef-4908-b7ec-6da4281fdc41)
![image](https://github.com/user-attachments/assets/604c446a-dfde-4503-abf4-80d4418572d5)
![image](https://github.com/user-attachments/assets/c912cb9b-7e95-4015-99da-50e7dc166d1d)
![image](https://github.com/user-attachments/assets/099d6a4c-6a0f-4b4d-98b9-6e53e5773560)
![image](https://github.com/user-attachments/assets/af2fe46a-2b75-4504-b697-1ddd320f1cc1)
![image](https://github.com/user-attachments/assets/7fb72215-fe46-49f8-aa31-303e5f3cd8d9)


### RESULT:
Thus, the SARIMA model was successfully implemented, accurately forecasting Google stock prices with measurable error evaluation.
