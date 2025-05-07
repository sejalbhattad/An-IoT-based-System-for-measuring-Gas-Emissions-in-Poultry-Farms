import io
import os
import pandas as pd
import requests
from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)


READ_API_KEY = '9L0Z8GS9R0L1QQ5M'
CHANNEL_ID = '2923995'


model = joblib.load('gas_forecast_model.joblib')
scaler = joblib.load('gas_scaler.joblib')


def download_all_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=10000"
    r = requests.get(url)
    feeds = r.json()["feeds"]
    df = pd.DataFrame(feeds)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['Field 1'] = pd.to_numeric(df['field1'], errors='coerce')
    df['Field 2'] = pd.to_numeric(df['field2'], errors='coerce')
    df['Field 3'] = pd.to_numeric(df['field3'], errors='coerce')
    df = df[['created_at', 'Field 1', 'Field 2', 'Field 3']].dropna()
    if os.path.exists('data.csv'):
        os.remove('data.csv')
    df.to_csv('data.csv', index=False)
    return df

def predict_next_day(df):
    df = df[['created_at', 'Field 1', 'Field 2', 'Field 3']].dropna()
    df.columns = ['created_at', 'CO2', 'CH4', 'NH3']
    df.set_index('created_at', inplace=True)
    df_hourly = df.resample('1H').mean().dropna()

    expected_features = model.n_features_in_
    num_gases = 3
    window_size = expected_features // num_gases

    if len(df_hourly) < window_size:
        raise ValueError("Not enough data for prediction.")

    scaled_data = scaler.transform(df_hourly)
    current_input = scaled_data[-window_size:].flatten().reshape(1, -1)

    preds = []
    for _ in range(24):
        pred = model.predict(current_input)[0]
        preds.append(pred)
        current_input = np.append(current_input.flatten()[num_gases:], pred).reshape(1, -1)

    preds_unscaled = scaler.inverse_transform(preds)
    last_time = df_hourly.index[-1]
    future_time = [last_time + pd.Timedelta(hours=i + 1) for i in range(24)]

    pred_df = pd.DataFrame(preds_unscaled, columns=['CO2', 'CH4', 'NH3'], index=future_time)
    return pred_df



@app.route('/')
def dashboard():
    df = download_all_data()
    latest = df.iloc[-1]
    today_str = datetime.now().strftime("%A, %d %B %Y - %I:%M %p")
    graph_data = df.to_json(orient='records')
    return render_template('dashboard.html', latest=latest, today_str=today_str, graph_data=graph_data, prediction=None)

@app.route('/predict')
def predict():
    df = download_all_data()
    pred_df = predict_next_day(df)
    pred_data = pred_df.reset_index().to_json(orient='records')

    df = download_all_data()
    latest = df.iloc[-1]
    today_str = datetime.now().strftime("%A, %d %B %Y - %I:%M %p")
    graph_data = df.to_json(orient='records')

    return render_template('dashboard.html', latest=latest, today_str=today_str, graph_data=graph_data, prediction=pred_data)

@app.route('/gas/<selected_gas>', methods=['GET', 'POST'])
def gas_data(selected_gas):
    df = download_all_data()

    gas = selected_gas
    from_date = None
    to_date = None
    period = 'all'
    filtered_df = pd.DataFrame()

    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        period = request.form.get('period', 'all')

        if from_date and to_date:
            from_date = pd.to_datetime(from_date)
            to_date = pd.to_datetime(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            df = pd.read_csv('data.csv', parse_dates=['created_at'])
            df['created_at'] = df['created_at'].dt.tz_localize(None)

            df = df[(df['created_at'] >= from_date) & (df['created_at'] <= to_date)]

            gas_col = {'CO2': 'Field 1', 'CH4': 'Field 2', 'NH3': 'Field 3'}[gas]
            df = df[['created_at', gas_col]].rename(columns={gas_col: 'value'})

            df['hour'] = df['created_at'].dt.hour
            if period == 'day':
                df = df[df['hour'].between(9, 12)]
            elif period == 'evening':
                df = df[df['hour'].between(13, 18)]
            elif period == 'night':
                df = df[(df['hour'] >= 19) | (df['hour'] < 9)]

            filtered_df = df.drop(columns='hour')

    return render_template('gas_data.html', gas=gas, filtered=filtered_df.to_json(orient='records'),
                           from_date=from_date.strftime('%Y-%m-%d') if from_date else None,
                           to_date=to_date.strftime('%Y-%m-%d') if to_date else None,
                           period=period)

@app.route('/download_gas', methods=['POST'])
def download_gas():
    gas = request.form.get('gas')
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')
    period = request.form.get('period', 'all')

    df = pd.read_csv('data.csv', parse_dates=['created_at'])
    df['created_at'] = df['created_at'].dt.tz_localize(None)

    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df = df[(df['created_at'] >= from_date) & (df['created_at'] <= to_date)]

    gas_col = {'CO2': 'Field 1', 'CH4': 'Field 2', 'NH3': 'Field 3'}[gas]
    df = df[['created_at', gas_col]].rename(columns={gas_col: f'{gas} (ppm)'})

    df['hour'] = df['created_at'].dt.hour
    if period == 'day':
        df = df[df['hour'].between(9, 12)]
    elif period == 'evening':
        df = df[df['hour'].between(13, 18)]
    elif period == 'night':
        df = df[(df['hour'] >= 19) | (df['hour'] < 9)]

    df = df.drop(columns='hour')

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype='text/csv',
                     download_name=f'{gas}_filtered.csv',
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
