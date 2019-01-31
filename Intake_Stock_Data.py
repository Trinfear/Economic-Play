#!python3

'''

get list of sp500 tickers from wikipedia
load all stock data from yahoo
save all data to a new file

'''


import pickle
import requests
import bs4 as bs
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web


start_time = dt.datetime(2012, 1, 1)
end_time = dt.datetime.now()
save_file_dir = 'SP500 Data'


def get_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    return tickers


def get_stock_data(tickers, start, end, save_dir):

    for ticker in tickers:     # TODO: figure out how many stock data sets you want with tickers[:x]:
        print(ticker)
        # might need to add in something here to recheck data daily if need be
        df = web.DataReader(ticker, 'yahoo', start, end)
        # do more to clean up data
        # remove any points where value is zero?
        # remove any points preceding a break?
        for day in df.iterrows():
            # check if date is one day after previous?
            # check if value isn't 0
            # if either, drop the row
            if day[1][1] == 0:
                df.drop(day)
            pass
        df.to_csv('{}/{}.csv'.format(save_dir, ticker))


def update_stocks(tickers):
    # iterate through tickers
    # find the end date of its stock count
    # get stock data from previous end date to now
    # append to stock file

    # load csv
    # find last date
    # use last date as start
    # use current time and end
    # get new data
    # append new data to loaded csv
    # resave csv
    pass


if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    get_stock_data(sp500_tickers, start_time, end_time, save_file_dir)
