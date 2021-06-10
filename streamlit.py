#!/usr/bin/env python
# coding: utf-8
from datetime import date
import streamlit as st
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Prediction')

stock_name=("ADANIPOWER.NS","BANKINDIA.NS")
selected_stock = st.selectbox('Select your Stock', stock_name)

month = st.slider(' Monthly Prediction:', 1,12 )
period = month*30 

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Please wait...')
data = load_data(selected_stock)
data_load_state.text('Done.. !!')

st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Chart', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(seasonality_mode='multiplicative',
daily_seasonality = True,
weekly_seasonality= True,
yearly_seasonality = True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
st.subheader("Predicted Stock price ")
st.write(forecast[['ds','yhat']].tail(1))

st.subheader('Forecast for Months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)







