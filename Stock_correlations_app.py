# # --- # Import statements

import yfinance as yf

import scipy
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import time
import re

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Made by Charles Ashton, M.S.'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# # --- # Sidebar elements

st.sidebar.subheader('Data retrieval parameters')


tickers = st.sidebar.text_input(label = 'Enter two or more tickers, separated with a space, e.g., \'aapl TSLA Msft\' ')

if tickers:
    
    x = re.search(r'[^a-zA-Z$]|[ ][ ]', tickers)

    if x:

        st.sidebar.warning('Enter only valid tickers separated by a single space', icon="⚠️")


tickers = tickers.upper()
ticker_list = tickers.split(" ")

indexlist = ('', '^IXIC', '^DJI', '^RUT')

index = st.sidebar.selectbox(label='Select a benchmark index for correlation: \n (IXIC = Nasdaq Composite; DJI = Dow Jones Industrial Average; RUT = Russell 2000)',
    options = indexlist,
    format_func=lambda x: 'Select appropriate index' if x == '' else x)

N = st.sidebar.slider(label = 'Select number of calendar days to include in look-back period', min_value=30,
                      max_value=360, value=90, step=30)

date_N_days_ago = datetime.now() - timedelta(days=N)
start = date_N_days_ago.strftime('%Y-%m-%d')
today = date.today().strftime('%Y-%m-%d')

start_disp = date_N_days_ago.strftime('%d %B, %Y')
today_disp = date.today().strftime('%d %B, %Y')

st.sidebar.write(f'The selected look-back period spans \n {start_disp} to {today_disp}.')


# # --- # Function definitions

def stock_data():

    stock_data = yf.download(f'{tickers} {index}', start=start, end=today)

    AdjClose = stock_data.loc[:, 'Adj Close']

    AdjClose.rename_axis(index=None, columns=None, inplace=True)

    # st.write(AdjClose) # checkpoint; may delete in final script

    return AdjClose


def day_func(x):

    trading_days = len(x.index)

    return st.write(f'Pearson\'s R correlation matrix for the {trading_days} market trading days between {start_disp} and {today_disp}.')


def correlogram_func(x):

    pct_returns = x.pct_change() # 1-day percent changes in price

    corr_matrix = pct_returns.corr() # get correlation matrix

    mask = np.zeros_like(corr_matrix) # return an array of zeros

    triangle_indices = np.triu_indices_from(mask) # return the indices for the upper-triangle of mask

    mask[triangle_indices] = True # Set the upper-triangle indices to '1' (True)

    fig, ax = plt.subplots(figsize=(3,3))

    ax = sns.heatmap(ax=ax, data=corr_matrix, 
                 mask=mask, 
                 annot=True, 
                 linewidth=.5, cmap="crest", 
                 annot_kws={"size": 7})
    ax.set_title(f'Correlogram')
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.tick_params(axis='y', labelrotation=0, labelsize=7)

    return st.pyplot(fig)

# # --- # Correlation line plot

def pairwiseR_func(x):

    df_change = x.pct_change()

        # st.write(df_change) # checkpoint; may delete from final script

    for i in ticker_list:

        df_change[i] = df_change[index].rolling(window_size).corr(df_change[i])

    # st.write(df_change) # checkpoint; may delete from final script

    df_change.drop(f'{index}', axis=1, inplace=True)
    df_change.dropna(inplace=True)

    mean_R = df_change.values.mean() # computes the mean R value across all tickers

    # st.write(df_change) # checkpoint; may delete from final script

    fig = plt.figure(figsize=(3,3))

    ax = fig.subplots()

    ax.plot(df_change, alpha=0.7)

    ax.set_ylabel('Pearson\'s R')
    ax.set_title(f'21 day rolling correlation with {index}')
    ax.tick_params(axis='x', which='major', labelsize=7, labelrotation=45)
    ax.tick_params(axis='y', which='major', labelsize=7, labelrotation=0)
    ax.axhline(mean_R, color='blue', alpha=0.7)
    ax.legend(df_change.columns.tolist())

    return st.pyplot(fig)

# # --- # Main body

st.title("Stock Price Correlations")

col1, col2 = st.columns((1, 1))

with col1:

    if len(ticker_list) >= 2 and index != '':

        stock_data()

        day_func(stock_data())

    else: st.write('Please choose parameters on the left')

with col2:

    if len(ticker_list) >= 2 and index != '':

        window_size = st.slider('Choose number of trading days for rolling-window correlation (Pearson\'s R)', 
            min_value=7, max_value=28, value=21, step=7)

    else: st.write('')

col3, col4 = st.columns((1, 1))

with col3:

    if len(ticker_list) >= 2 and index != '':

        with st.spinner('Working...'):
            time.sleep(2)

        stock_data()

        correlogram_func(stock_data())

with col4:

    if len(ticker_list) >= 2 and index != '':

        with st.spinner('Working...'):
            time.sleep(2)

        pairwiseR_func(stock_data())

        st.write('Note: horizontal blue line represents mean R value across indicated dates')








