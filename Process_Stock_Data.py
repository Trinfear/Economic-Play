#!python3
# go through early financial tutorials with sentdex and find predictive analysis
# use that as a base and go with it
# try to throw in some machine learning?
    # give neural network a series of prices and have it guess future prices?
    # use clustering to pick stocks?
        # figure out a good blend of stocks based on things that rise or fall against eachother?
        # this way even if one falls, the other will rise for overall gain
        # try to find stocks similar to those who have ballooned and pick them up before they do?
    # make a decision tree to figure out if you should buy a stock based on previous stock prices?
    #


import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc


data_dir = 'SP500 Data'


def load_stock(stock_dir):
    # load a specific stock
    df = pd.DataFrame.from_csv(stock_dir)
    return df


def load_all_data(load_data_dir):
    data = []
    for file in os.listdir(load_data_dir):
        load_dir = load_data_dir + '/' + file
        df = pd.DataFrame.from_csv(load_dir)
        # this gives depreciation warning, but using read_csv doesn't work for candlestick graphs
        name = file[:-4]
        data.append((name, df))
    return data


def create_sp500_map(sp_list):
    # intake data and create a single sp500 df
    sp_df = pd.DataFrame()
    for stock in sp_list:
        name = stock[0]
        df = stock[1].copy()
        df.rename(columns={'Adj Close': name}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if sp_df.empty:
            sp_df = df
        else:
            sp_df = sp_df.join(df, how='outer')

    sp_avg = pd.DataFrame(sp_df.mean(axis=1))

    # how to get average stock value for each day?
    return sp_df, sp_avg


def heat_map(df):
    df_corr = df.corr()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()   # why?
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


def group_similar(df, ticker, n_count=10):
    # get stocks which rise and fall with positive correlation
    # currently also includes the ticker as closest value
    df_corr = df.corr()
    df_corr = df_corr[ticker]

    # for some reason ascending=false returns highest values?
    df_corr = df_corr.sort_values(ascending=False)
    closest = df_corr[:n_count]
    
    return closest


def group_opposed(df, ticker, n_count=10):
    # get stocks which rise and fall with negative correlation
    # currently also includes the ticker as closest value
    df_corr = df.corr()
    df_corr = df_corr[ticker]

    # for some reason ascending=false returns highest values?
    df_corr = df_corr.sort_values()
    farthest = df_corr[:n_count]
    
    return farthest


def graph_stock(df, tick_space=100):
    # graph a stock over time, and make a simple future prediction
    data_size = len(df)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(df)
    
    ax1.xaxis_date()

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.legend(labels=range(10))
    plt.show()


def candlestick_graph(df, tick_space=100):
    # must intake a full stock dataframe for a single company

    data_size = len(df)
    
    df_ohlc = df.resample('10D').ohlc()
    df_ohlc = df_ohlc.reset_index()
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.xaxis_date()

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    candlestick_ohlc(ax1, df_ohlc.values, width=1, colorup='g')

    plt.show()

##     while x < y:
##         append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
##         ohlc.append(append_me)
##         x+=1
## candlestick_ohlc(ax1, ohlc)


def get_neg_corr(df, count=10):
    split = int(len(df) * 0.6)
    new_df = df[:split]
    
    df_corr = new_df.corr()
    closest = []
    indexes = df_corr.columns.values
    for column in df_corr:
        index = 0
        for value in df_corr[column]:
            # should initial tickers/corr be the ones closest to zero or most negative
            closest.append((value, (column, indexes[index])))
            closest.sort()
            closest = closest[:10]
            index += 1

    tickers = []
    corr = closest[0][0]
    
    tickers.append(closest[0][1][0])
    tickers.append(closest[0][1][1])

    print(tickers, corr)
    # iterate through corr for each tickers, and assign each ticker a value for sum of corr
    # pick ticker with lowest value
    # continue until you have count tickers
    while len(tickers) < count:
        new_corrs = {}
        for ticker in tickers:
            # iterate through each corr value and
            index = 0
            for value in df_corr[ticker]:
                if indexes[index] in new_corrs.keys():
                    new_corrs[indexes[index]] += value
                else:
                    new_corrs[indexes[index]] = value
                index += 1

        # add in a check to make sure its not in the list
        new_ticker = min(new_corrs, key = new_corrs.get)
        corr_value = new_corrs[new_ticker]
        del new_corrs[new_ticker]
        
        while new_ticker in tickers:
            new_ticker = min(new_corrs, key = new_corrs.get)
            corr_value = new_corrs[new_ticker]
            del new_corrs[new_ticker]

        tickers.append(new_ticker)
        corr += corr_value

    print(tickers, corr)

    graph_df = pd.DataFrame(df[tickers[0]])
    for ticker in tickers[1:]:
        graph_df = graph_df.join(df[ticker], how='outer')

    df_avg = pd.DataFrame(graph_df.mean(axis=1))
    # print(df_mean[0], df_mean[-1])
    # print(sp500_avg[0][0], sp500_avg[0][-1])

    # df_avg['sp_avg'] = sp500_avg
    graph_stock(df_avg)
    # graph mean of this stock compared to mean of full group
    
    return tickers, corr, df_avg


def get_no_corr(df, count=10):
    split = int(len(df) * 0.6)
    new_df = df[:split]

    df_corr = new_df.corr()
    closest = []
    indexes = df_corr.columns.values
    for column in df_corr:
        index = 0
        for value in df_corr[column]:
            # should initial tickers/corr be the ones closest to zero or most negative
            dist = np.absolute(value)
            closest.append((dist, (column, indexes[index])))
            closest.sort()
            closest = closest[:10]
            index += 1

    tickers = []
    corr = closest[0][0]
    
    tickers.append(closest[0][1][0])
    tickers.append(closest[0][1][1])

    print(tickers, corr)
    # iterate through corr for each of the tickers
    # find the tickers which bring the corr closest to 0
    
    while len(tickers) < count:
        new_corrs = {}
        for ticker in tickers:
            # iterate through each corr value and
            index = 0
            for value in df_corr[ticker]:
                dist = np.absolute(value)
                if indexes[index] in new_corrs.keys():
                    new_corrs[indexes[index]] += dist
                else:
                    new_corrs[indexes[index]] = dist
                index += 1

        
        new_ticker = min(new_corrs, key = new_corrs.get)
        corr_value = new_corrs[new_ticker]
        del new_corrs[new_ticker]
        
        while new_ticker in tickers:
            new_ticker = min(new_corrs, key = new_corrs.get)
            corr_value = new_corrs[new_ticker]
            del new_corrs[new_ticker]

        tickers.append(new_ticker)
        corr += corr_value
        
    print(tickers, corr)

    graph_df = pd.DataFrame(df[tickers[0]])
    for ticker in tickers[1:]:
        graph_df = graph_df.join(df[ticker], how='outer')

    df_avg = pd.DataFrame(graph_df.mean(axis=1))
    # print(df_mean[0], df_mean[-1])
    # print(sp500_avg[0][0], sp500_avg[0][-1])

    # df_avg['sp_avg'] = sp500_avg
    graph_stock(df_avg)
    # graph mean of this stock compared to mean of full group
    
    return tickers, corr, df_avg


def get_pos_corr(df, count=10):
    # this is a risky investment, it moves with the specific group of stocks
    # add values of corrs for stocks to get the highest corr value

    # start with stock with most total improvement
    # get stocks with most positive correlations that aren't already included
    split = int(len(df) * 0.6)
    new_df = df[:split]
    # this is still too close, to be in sp500 we know the stocks ended up doing well

    changes = {}
    for column in new_df:
        # total change or percent change?
        start = new_df[column][0]
        end = new_df[column][-1]
        change = end - start
        changes[column] = change
    
    tickers = []
    
    new_ticker = max(changes, key = changes.get)
    tickers.append(new_ticker)
    del changes[new_ticker]

    new_ticker = max(changes, key = changes.get)
    tickers.append(new_ticker)
    del changes[new_ticker]

    df_corr = new_df.corr()
    indexes = df_corr.columns.values
    corr = df_corr[tickers[0]][tickers[1]]
    
    while len(tickers) < count:
        new_corrs = {}
        for ticker in tickers:
            # iterate through each corr value and
            index = 0
            for value in df_corr[ticker]:
                if indexes[index] in new_corrs.keys():
                    new_corrs[indexes[index]] += value
                else:
                    new_corrs[indexes[index]] = value
                index += 1

        # add in a check to make sure its not in the list
        new_ticker = max(new_corrs, key = new_corrs.get)
        corr_value = new_corrs[new_ticker]
        del new_corrs[new_ticker]
        
        while new_ticker in tickers:
            new_ticker = max(new_corrs, key = new_corrs.get)
            corr_value = new_corrs[new_ticker]
            del new_corrs[new_ticker]

        tickers.append(new_ticker)
        corr += corr_value

    print(tickers, corr)

    graph_df = pd.DataFrame(df[tickers[0]])
    for ticker in tickers[1:]:
        graph_df = graph_df.join(df[ticker], how='outer')

    df_avg = pd.DataFrame(graph_df.mean(axis=1))
    # print(df_mean[0], df_mean[-1])
    # print(sp500_avg[0][0], sp500_avg[0][-1])

    # df_avg['sp_avg'] = sp500_avg
    graph_stock(df_avg)
    # graph mean of this stock compared to mean of full group
    
    return tickers, corr, df_avg


def good_investment():
    # intake an amount of money and buy stocks using one of the above strategies
    pass


def best_plan(df):
    # run all three corr
    # get mean of each
    # graph them and print start and end and perceny improvement
    # rewrite the graphings within the function and change the returns

    # limit the check range to the first three years, but add the last two to the average to see how it did

    tickers_1, corr_1, graph_1 = get_pos_corr(df, count=10)
    tickers_2, corr_2, graph_2 = get_no_corr(df, count=10)
    tickers_3, corr_3, graph_3 = get_neg_corr(df, count=10)

    graph_df = graph_1

    graph_df['two'] = graph_2
    graph_df['three'] = graph_3
    graph_df['avg'] = sp500_avg

    graph_stock(graph_df)
    pass


# iterate through each day and add high, low, open and close  an ohlc object and graph it


if __name__ == '__main__':

    data_set = load_all_data(data_dir)
    print('loaded')
    
    sp500_df, sp500_avg = create_sp500_map(data_set)
    print('consolidated')

    best_plan(sp500_df)

    graph_stock(sp500_avg)
    
    candlestick_graph(sp500_avg)








