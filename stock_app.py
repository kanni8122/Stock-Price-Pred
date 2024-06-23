import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

indian_stocks = {
    'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys',
    'RELIANCE.NS': 'Reliance Industries',
    'HDFCBANK.NS': 'HDFC Bank',
    'HDFC.NS': 'Housing Development Finance Corporation',
    'ICICIBANK.NS': 'ICICI Bank',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'LT.NS': 'Larsen & Toubro',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'AXISBANK.NS': 'Axis Bank',
    'HCLTECH.NS': 'HCL Technologies',
    'TITAN.NS': 'Titan Company',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'WIPRO.NS': 'Wipro',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'NESTLEIND.NS': 'Nestle India',
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'POWERGRID.NS': 'Power Grid Corporation',
    'ADANIPORTS.NS': 'Adani Ports & SEZ',
    'HINDALCO.NS': 'Hindalco Industries',
    'TATAMOTORS.NS': 'Tata Motors',
    'DIVISLAB.NS': 'Divi\'s Laboratories',
    'JSWSTEEL.NS': 'JSW Steel',
    'M&M.NS': 'Mahindra & Mahindra',
    'TECHM.NS': 'Tech Mahindra',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'BPCL.NS': 'Bharat Petroleum Corporation',
    'ONGC.NS': 'Oil and Natural Gas Corporation',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'BRITANNIA.NS': 'Britannia Industries',
    'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India',
    'GRASIM.NS': 'Grasim Industries',
    'SBILIFE.NS': 'SBI Life Insurance',
    'TATASTEEL.NS': 'Tata Steel',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'EICHERMOT.NS': 'Eicher Motors',
    'SHREECEM.NS': 'Shree Cement',
    'TATACONSUM.NS': 'Tata Consumer Products',
    'NTPC.NS': 'NTPC Limited',
    'UPL.NS': 'UPL Limited',
    'HDFCLIFE.NS': 'HDFC Life Insurance'
}

global_stocks = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet (Google)',
    'AMZN': 'Amazon',
    'META': 'Meta Platforms (Facebook)',
    'TSLA': 'Tesla',
    'BRK-B': 'Berkshire Hathaway',
    'V': 'Visa',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart',
    'JPM': 'JPMorgan Chase',
    'MA': 'Mastercard',
    'PG': 'Procter & Gamble',
    'UNH': 'UnitedHealth Group',
    'DIS': 'Disney',
    'NVDA': 'NVIDIA',
    'HD': 'Home Depot',
    'PYPL': 'PayPal',
    'VZ': 'Verizon',
    'ADBE': 'Adobe',
    'NFLX': 'Netflix',
    'CMCSA': 'Comcast',
    'INTC': 'Intel',
    'KO': 'Coca-Cola',
    'PEP': 'PepsiCo',
    'T': 'AT&T',
    'CSCO': 'Cisco Systems',
    'ABT': 'Abbott Laboratories',
    'MRK': 'Merck & Co.',
    'XOM': 'ExxonMobil',
    'PFE': 'Pfizer',
    'NKE': 'Nike',
    'MCD': 'McDonald\'s',
    'IBM': 'IBM',
    'CRM': 'Salesforce',
    'BA': 'Boeing',
    'WBA': 'Walgreens Boots Alliance',
    'MMM': '3M',
    'ORCL': 'Oracle',
    'ACN': 'Accenture',
    'CVX': 'Chevron',
    'MDT': 'Medtronic',
    'HON': 'Honeywell',
    'COST': 'Costco',
    'NEE': 'NextEra Energy',
    'TXN': 'Texas Instruments',
    'PM': 'Philip Morris International',
    'LLY': 'Eli Lilly',
    'AMGN': 'Amgen'
}

all_stocks = {**indian_stocks, **global_stocks}

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, data, training_size=0.8):
    training_data_len = int(np.ceil(len(data) * training_size))
    train_data = data[0:training_data_len]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model, training_data_len

def make_predictions(model, data, training_data_len):
    test_data = data[training_data_len - 60:, :]
    x_test = []
    y_test = data[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    return predictions, y_test

def main():
    st.title('*Stock Price Prediction System*')

    selected_stock = st.selectbox('Select Stock', list(all_stocks.keys()), format_func=lambda x: all_stocks[x])

    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

    if st.button('Fetch Data'):
        data = fetch_data(selected_stock, start_date, end_date)
        st.subheader("Data of the Selected Stock " + selected_stock)
        st.write(data.tail())
        st.divider()

        if len(data) < 60:
            st.error("Not enough data to create sequences. Please select a longer date range.")
            return

        scaled_data, scaler = preprocess_data(data)
        model = create_model()
        model, training_data_len = train_model(model, scaled_data)

        predictions, y_test = make_predictions(model, scaled_data, training_data_len)
        predictions = scaler.inverse_transform(predictions)

        actual_prices = data['Close'][training_data_len:].values
        prediction_data = pd.DataFrame({
            'Actual': actual_prices,
            'Predictions': predictions.flatten()
        }, index=data.index[training_data_len:])
        st.subheader('Prediction Data')
        st.line_chart(prediction_data)
        st.divider()
        st.dataframe(prediction_data)
        st.divider()
        st.subheader("Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        st.plotly_chart(fig)
        st.divider()

        # Moving averages
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-Day MA'))
        ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200-Day MA'))
        st.subheader("Moving Averages")
        st.plotly_chart(ma_fig)

        #st.subheader("Metrics")
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2=r2_score(y_test,predictions)

        with st.expander("Metrics"):

            st.write(f"R2 Score : {r2}")
            st.write(f"Root Mean Squared Error: {rmse}")
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")

        csv = prediction_data.to_csv().encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", key='download-csv')

if __name__ == "__main__":
    main()
