from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from data_helpers import dateparser
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import OrderedDict
import datetime
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error

def generate_x_axis(start_date, n_test=None):
  return pd.date_range(start_date, periods=n_test, freq='D')

vpi = 1
data = pd.read_csv("./data/production_series_"+str(vpi)+".csv")


data = data[["Date", "Oil_rate"]]

mask = (data["Date"] <= '2016-10-30') & (data["Date"] > '2014-06-13')
data = data.loc[mask]

data_backup = data

data = data[["Oil_rate"]].values

step_preds_5 = {}
step_preds_1 = {}
step_preds_prophet = {}
step_preds_nn = {}

error = {'5': [], '1': [], 'prophet': [], 'nn': []}


n_test = 30
for step in range(60, len(data), n_test):
  
  train = data[0:step]
  test = data[step:step+n_test]

  print step

  try:
    model = ARIMA(train, order=(5, 1, 5))
    model = model.fit(disp=0)

    preds = model.predict(start=len(train), end=len(train)+n_test-1, typ="levels")
    step_preds_5[step] = preds
    error['5'].append(np.sqrt(mean_squared_error(test, step_preds_5[step])))
  except:
    pass


  try:
    model = ARIMA(train, order=(5, 1, 1))
    model = model.fit(disp=0)

    preds = model.predict(start=len(train), end=len(train)+len(test)-1, typ="levels")
    step_preds_1[step] = preds
    error['1'].append(np.sqrt(mean_squared_error(test, step_preds_1[step])))
  except:
    pass

  #feed forward
  #====================================================================
  train_data = np.reshape(train, len(train)).tolist()
  test_data = np.reshape(test, len(test)).tolist()
  window_input = 30
  window_output = 1

  x = []
  y = []


  for i in range(len(train_data)-window_input-1):
    x.append(train_data[i:i+window_input])
    y.append(train_data[i+1:i+window_input+1])

  x_train = np.array(x)
  y_train = np.array(y)

  # x_train = np.reshape(x_train, (x_train.shape[0], window_input, 1))

  model = Sequential()
  model.add(Dense(16, input_dim=window_input, kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(16, kernel_initializer='normal', activation='relu'))
  model.add(Dense(window_input, kernel_initializer='normal'))
  model.compile(loss="mean_squared_error", optimizer="adam")

  model.fit(x_train, y_train, epochs=200, batch_size=32, shuffle=True, verbose=0)

  preds = train_data[-window_input:]
  for i in range(0, len(test_data)):
    input = np.reshape(preds[-window_input:], (1, window_input))
    pred = model.predict(input).tolist()[0][-1]
    preds.append(pred)
    
  step_preds_nn[step] = preds[window_input:]
  error['nn'].append(np.sqrt(mean_squared_error(test, step_preds_nn[step])))


  #prophet
  #====================================================================
  playoff_dates = data_backup[data_backup.Oil_rate == 0]["Date"].values
  holidays = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(playoff_dates),
    'lower_window': 0,
    'upper_window': 1,
  })

  train = data_backup.iloc[0:step]
  test = data_backup.iloc[step:step+n_test]
  train.columns = ["ds", "y"]
  test.columns = ["ds", "y"]

  test = test[["y"]].values
  m = Prophet(changepoint_range=1, interval_width=0.7, holidays=holidays)
  # m.add_regressor('regressor', mode='additive')
  m.fit(train)

  future = pd.date_range(datetime.datetime.strptime(train.iloc[-1, 0], '%Y-%m-%d')+ datetime.timedelta(days=1), periods=len(test), freq='D')
  future = pd.DataFrame({'ds': future})

  preds = m.predict(future)


  yhats = preds[['yhat']].values
  for i in range(len(yhats)):
    if yhats[i] < 0:
      yhats[i] = 0

  yhats = np.reshape(yhats, len(yhats))

  step_preds_prophet[step] = yhats
  error['prophet'].append(np.sqrt(mean_squared_error(test, step_preds_prophet[step])))

for key in error.keys():
  print key, ': ', np.mean(error[key]), error[key]
print "Length: ", len(data)

start_date = data_backup.iloc[0, 0]

#plot
fig = plt.figure(figsize=(16, 8))
plt.plot(generate_x_axis(start_date, len(data)), data, color="red", label="Real", linewidth=2)
for key in step_preds_5.keys():
  plt.plot(generate_x_axis(data_backup.iloc[key, 0], len(step_preds_5[key])), step_preds_5[key], color="blue", label="ARIMA(5,1,5)", linewidth=2)
for key in step_preds_1.keys():
  plt.plot(generate_x_axis(data_backup.iloc[key, 0], len(step_preds_1[key])), step_preds_1[key], color="green", label="ARIMA(5,1,1)", linewidth=2)
  plt.axvline(data_backup.iloc[key, 0], linestyle="-")
for key in step_preds_prophet:
  plt.plot(generate_x_axis(data_backup.iloc[key, 0], len(step_preds_prophet[key])), step_preds_prophet[key], color="Orange", label="Prophet", linewidth=2)
for key in step_preds_nn:
  plt.plot(generate_x_axis(data_backup.iloc[key, 0], len(step_preds_nn[key])), step_preds_nn[key], color="Purple", label="DNN", linewidth=2)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="lower left")
# plt.legend(loc="upper right", numpoints=1)
# plt.savefig("./results/"+str(vpi)+".png")
plt.show()
