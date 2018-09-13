# Forecasting the production of oil

## Combining models:

- There are three models using in this method: ARIMA, Prophet and Simple DNN.
- ARIMA includes ARIMA(5,1,1) and ARIMA(5,1,5)
- Using three parameters(changepoint_range, interval_width, holidays) to tune this model
- Simple DNN has two layers, each layer contains 16 neurons. This network is trained by using Adam optimizer, batch training.
- Using RMSE(No smoothing the predictions or eliminating outliers) and visualization to evaluate the performance of three models.

## Pyflux

- It is a package for forecasting time series. It provides a number of methods such as ARIMA or GARCH

## XGBoost vs ARIMA

- In this section, we focus on looking for the effective ways to generate features from time series for XGB models.
- Two types of prediction(predicting one time and predicting step to step) were also implemented
- Using ARIMA(2,1,1) to compare the performance of XGB models.
- The ARIMA parameters were selected by using ACF and PACF.
- Using simple method(z-score, IQR) to detect and eliminate outliers.
- Before calculating RMSE, the time series was smoothed by convolutional fitters.
- Using RMSE(Smoothing the predictions and elimating outliers) and visualization to evaluate the performance of three models.
