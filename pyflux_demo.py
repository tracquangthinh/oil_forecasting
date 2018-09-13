import pandas as pd
from data_helpers import dateparser, median_smooth
from matplotlib import pyplot as plt
import numpy as np
import random
import pyflux as pf

def generate_x_axis(start_date, n_test=None):
  return pd.date_range(start_date, periods=n_test, freq='D')

vpi = 1
n_test = 30
data = pd.read_csv("./data/production_series_"+str(vpi)+".csv")

data = data[["Date", "Oil_rate"]]

mask = (data["Date"] > '2011-08-22') & (data["Date"] < '2012-03-19')
data = data.loc[mask]

data_backup = data

data = data[["Oil_rate"]]
data = data.fillna(0)
# data = data[data>500]

data.hist()
plt.show()

data = np.log(data["Oil_rate"]+1)

data = data.values

train_data = data[0:-n_test]
test_data = data[-n_test:]


# model = pf.GARCH(p=1, q=0, data=train_data)
# model = pf.ARIMA(data=train_data, ar=10, ma=5, family=pf.Normal())
# model = pf.DAR(data=train_data, ar=9, integ=1)
# model = pf.EGARCH(train_data, p=1, q=1)
model = pf.GAS(data, ar=2, sc=1, integ=1, family=pf.Normal())
result = model.fit('M-H')
print(result.summary())
preds = model.predict(h=n_test)

preds = preds.values
print preds

plt.plot(np.exp(test_data), color="red", label="Real", linewidth=2)
plt.plot(np.exp(preds) , color="green", label="Real", linewidth=2)
plt.show()