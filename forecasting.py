import pandas as pd
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import os
import wget
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date

def download_data():
    # Get data updated
    if os.path.exists('data.csv'):
        os.remove('data.csv')

    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
    try:
        file = wget.download(url, out='data.csv')
        print("\n")
    except:
        print("BAD", url)

def load_series():
    # Load series to predict
    series = pd.read_csv('data.csv')
    new_positives = series['nuovi_positivi'].values
    intensive_care = series['terapia_intensiva'].values
    total_positives = series['totale_positivi'].values
    dates = series['data'].values
    return new_positives, intensive_care, total_positives, dates

def ARIMA_model(series, order, days = 7):
    # Fitting and forecast the series
    train = [x for x in series]
    model = ARIMA(train, order = order)
    model_fit = model.fit(disp=0)
    forecast, err, ci = model_fit.forecast(steps = days, alpha = 0.05)
    start_day = date.today() + datetime.timedelta(days = 1)
    predictions_df = pd.DataFrame({'Forecast':forecast.round()}, index=pd.date_range(start = start_day, periods=days, freq='D'))
    return predictions_df, ci

def plot_results(series, df_forecast, ci, label):
    start_covid_day = date(2020, 2, 24)
    series = pd.DataFrame({'Real data':series}, index=pd.date_range(start = start_covid_day, periods=series.shape[0], freq='D'))
    ax = series.plot(label = 'Real Data', figsize = (20, 15))
    df_forecast.plot(ax = ax, label='Forecast', color = 'r')
    ax.fill_between(df_forecast.index,
                    ci[:,0],
                    ci[:,1], color='b', alpha=.25)
    ax.set_xlabel('Days')
    ax.set_ylabel(label)
    ax.set_title(label + ' Forecasting')
    plt.legend()
    plt.savefig('plots/' + label + '.png')
    #plt.show()

# Order for ARIMA model
order = {
    'new_positives': (3, 1, 0),
    'intensive_care': (3, 2, 2),
    'total_positives': (1, 2, 0)
}

# Dowload and loading series
download_data()
new_positives, intensive_care, total_positives, dates = load_series()

# Stats of today
new_positives_today, intensive_care_today, total_positives_today, dates_today = new_positives[-1], intensive_care[-1], total_positives[-1], dates[-1]

# Forecasting with ARIMA models
new_positives_pred, new_positives_ci = ARIMA_model(new_positives, order['new_positives'])
intensive_care_pred, intensive_care_ci = ARIMA_model(intensive_care, order['intensive_care'])
total_positives_pred, total_positives_ci = ARIMA_model(total_positives, order['total_positives'])

# Plot Results
plot_results(new_positives, new_positives_pred, new_positives_ci, 'New Positives')
plot_results(intensive_care, intensive_care_pred, intensive_care_ci, 'Intensive Care')
plot_results(total_positives, total_positives_pred, total_positives_ci, 'Total Positives')
