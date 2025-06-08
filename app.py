from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
import os
import logging
from urllib.parse import unquote  # ✅ Needed to decode special characters

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load data
def load_data():
    data_path = 'C:/Users/SAIF/Downloads/vegetable-price-analysis/data/kalimati_tarkari_dataset.csv'
    df = pd.read_csv(data_path)
    return df

@app.route('/')
def index():
    df = load_data()
    products = df['Commodity'].dropna().unique().tolist()
    return render_template('index.html', products=products)

@app.route('/api/products')
def get_products():
    df = load_data()
    products = df['Commodity'].dropna().unique().tolist()
    return jsonify(products)

# Fix for special characters like parentheses, spaces, slashes
@app.route('/api/price_data/<path:product>')
def get_price_data(product):
    product = unquote(product)  # decode %20, %28, etc.
    df = load_data()
    product_data = df[df['Commodity'] == product].copy()

    if product_data.empty:
        logging.warning(f"No data found for product: {product}")
        return jsonify({'dates': [], 'prices': []})

    product_data['Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
    product_data = product_data.dropna(subset=['Date'])

    product_data.set_index('Date', inplace=True)
    ts = product_data['Average'].resample('D').mean().ffill()

    return jsonify({
        'dates': ts.index.strftime('%Y-%m-%d').tolist(),
        'prices': ts.values.tolist()
    })

@app.route('/forecast/<path:product>', methods=['GET', 'POST'])
def forecast(product):
    product = unquote(product)
    df = load_data()
    product_data = df[df['Commodity'] == product].copy()

    if product_data.empty:
        return f"No data found for product: {product}", 404

    price_col = 'Average' if 'Average' in product_data.columns else 'Price'
    if price_col not in product_data.columns:
        return f"Column '{price_col}' not found.", 500

    product_data['Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
    product_data = product_data.dropna(subset=['Date'])
    product_data.set_index('Date', inplace=True)

    ts = product_data[price_col].resample('D').mean().ffill()

    model = ARIMA(ts, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=30)

    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, label='Historical Prices')
    plt.plot(forecast_dates, forecast, label='Forecast', color='red')
    plt.title(f'Price Forecast for {product}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    logging.debug(f"Generated plot_data length: {len(plot_data)}")

    return render_template('forecast.html', product=product, plot_url=plot_data)


if __name__ == '__main__':
    app.run(debug=True)
