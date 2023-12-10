import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

'''
This code reads price histories from csv files. it then applies a buying strategy
The final step outputs a new csv file that tells how much some one would make ( wins or losses)
given that trading strategy. This is simmilar to the concept of adding rewards over some policy. 


it needs to dump -1,0,1 
-1 = sell, 0 = hold, 1 = buy
buy sell strategies need to be messed with to account for the store of funds the user has

'''
# a buyer that buys by chance
def random_buyer(b0, dlow, mid, dhigh):
    b0[b0 <= mid - dlow] = -1
    b0[b0 >= mid + dhigh] = 1

    cond0 = b0 >= mid - dlow
    cond1 = b0 <= mid + dlow
    cond = cond0 & cond1
    b0[cond] = 0
    return b0

def buy_vol_mean(n, vol):
    b_ = np.zeros(n,)
    mean = vol.rolling(window = 30, min_periods=1, axis = 0).mean()
    mean[np.isnan(mean)]=0

    b_[vol>1.1*mean] = -1
    b_[vol<0.7*mean] = 1
    b_[mean == 0] = 0
    return b_

def buy_vol_imean(n, vol):
    b_ = np.zeros(n,)
    mean = vol.rolling(window = 30, min_periods=1, axis = 0).mean()
    mean[np.isnan(mean)]=0

    b_[vol>1.1*mean] = 1
    b_[vol<0.7*mean] = -1
    b_[mean == 0] = 0
    return b_

def buy_vol_open(n, vol, open):
    b_ = np.zeros(n,)
    vmean = vol.rolling(window = 30, min_periods=1, axis = 0).mean()
    vmean[-30::] = 0

    omean = open.rolling(window = 30, min_periods=1, axis = 0).mean()
    omean[-30::] = 0

    # conditions based on volume
    cond0 = vol > 1.1 * vmean
    cond1 = vol < 0.7 * vmean

    # conditions based on price
    cond2 = open > 1.1 * omean # Condition to check if price is high
    cond3 = (open >= 0.7 * omean) & (open <= 1.1 * omean) # check to see if the price is ok
    cond4 = open < 0.7 * omean

    # so here's how we are going to do it. stage 1 decide based on price
    b_[cond2] = -1 # if the price is high then I want to sell
    b_[cond4] = 1  # if the price is low I want to buy
    b_[cond3 & cond0] = 1# if the price is in a mid range look at the volume
    b_[cond3 & cond1] = -1# if the price is in a mid range look at the volume
    return b_

def VolCross(n, open):
    # strategy was to buy and sell on up cross and down cross on volume respectively.
    # BUY ON UPCROSSES in price, SELL on derivative of error(smol avg - big avg) going from positive to negative
    b_ = np.zeros(n,)
    # gold crossing is 200 day moving average against 50 day moving average
    # get the rolling average of the time points
    omean_200 = open.rolling(window = 200, min_periods=1, axis = 0).mean()
    omean_200[-200::] = 0 + omean_200.iloc[-200]

    # get the rolling average of the time points
    win2 = 10
    omean_50 = open.rolling(window = win2, min_periods=1, axis = 0).mean()
    omean_50[-win2::] = 0 + omean_50.iloc[-win2]

    # at only the points where we have crossing we care about a buy or a sell.
    cc = omean_50 > omean_200
    cc = cc.astype(int)
    # on down crossing you get a negative 1, on upcrossing you get a positive one
    cc = cc.diff()
    cc = cc.values
    cc[np.isnan(cc)] = 0
    # I want to sell on down crossings only, and buy at up crossings so this is b_
    b_ = 0 + cc
    return b_, omean_50, omean_200

def PriceCrosses(n, open):
    '''
    one strategy is to buy on golden crosses this is when the short average
    overtakes the long average this creates a positive error. when the error starts
    decreasing we sell beacause its probably a peak.

    so buy on upcrosses, sell on local area maxima
    another place to buy would be at local area minima
    so that is where Ill place buy signals next.
    '''
    b_ = np.zeros(n,)
    win_long = 200
    win_smol = 10
    #get the long and short windows rolling average
    omean_long = open.rolling(window = win_long, min_periods=1, axis = 0).mean()
    omean_long[-win_long::] = 0 + omean_long.iloc[-win_long]

    omean_smol = open.rolling(window = win_smol, min_periods=1, axis = 0).mean()
    omean_smol[-win_smol::] = 0 + omean_smol.iloc[-win_smol]

    # at only the points where we have crossing we care about a buy or a sell.
    err = omean_smol - omean_long
    b_ = 0 + err.values
    b_[np.isnan(b_)] = 0
    # cc = omean_smol > omean_long
    # cc = cc.astype(int)
    # # on down crossing you get a negative 1, on upcrossing you get a positive one
    # cc = cc.diff()
    # cc = cc.values
    # cc[np.isnan(cc)] = 0
    # # I want to sell on down crossings only, and buy at up crossings so this is b_
    # b_ = 0 + cc
    return b_, omean_smol, omean_long

def eval_cash(buy_strat, low, high, tag):
    #figure out the cash flow .
    money_low = -1*buy_strat*low
    money_high = -1*buy_strat*high
    print('strategy:{}'.format(tag))
    print('low total', sum(money_low))
    print('high total', sum(money_high))
    print('\n')
    return [sum(money_low), sum(money_high)]

if __name__ == '__main__':
    hdir = Path('stock_data/4538_7213_bundle_archive/SP500_wgurufeats')
    hdir2 = Path('stock_data/4538_7213_bundle_archive/')

    files_ = sorted(list(map(str,hdir.glob('*.csv'))))
    # print(files_)

    all_outs = []
    for ji, ff in enumerate(files_):
        ticky = ff.split('\\')[-1].strip('.csv')
        print(ticky)
        dat_ = pd.read_csv(ff)
        b0 = np.random.rand(dat_.shape[0],)

        # buy strategies
        dat_['b0'] = random_buyer(b0, 0.16, 0.5, 0.16)
        dat_['b1'] = buy_vol_mean(dat_.shape[0], dat_['Volume'])
        dat_['b2'] = buy_vol_imean(dat_.shape[0], dat_['Volume'])
        dat_['b3'] = buy_vol_open(dat_.shape[0], dat_['Volume'], dat_['Open'])
        dat_['b4'] = buy_vol_mean(dat_.shape[0], dat_['Open'])

        tags = ['random buy', 'buy_vol_mean', 'buy_vol_imean', 'buy_vol_open', 'buy_open_mean']
        ticks = [ticky for i in range(len(tags))]

        strats = ['b{}'.format(i) for i in range(5)]
        outs = np.array([eval_cash(dat_[st], dat_['Low'], dat_['High'], tt) for st,tt in zip(strats, tags)])
        lulz = {'ticker':ticks, 'strat':tags, 'low_tot':outs[:,0], 'high_tot':outs[:,1]}
        lulz = pd.DataFrame.from_dict(lulz)
        all_outs.append(lulz)
        all_outs_ = pd.concat(all_outs, axis = 0)
        all_outs_.to_csv(hdir2 / 'strategy_costs.csv')

    # on the last itteration I want to get the average value stuff
    ticks = ['average' for k in range(len(tags))]
    meas = [np.mean(all_outs_['low_tot'].values[jj::5]) for jj in range(5)]
    meas2 = [np.mean(all_outs_['high_tot'].values[jj::5]) for jj in range(5)]
    lulz = {'ticker':ticks, 'strat':tags, 'low_tot':meas, 'high_tot':meas2}
    lulz = pd.DataFrame.from_dict(lulz)
    all_outs.append(lulz)

    ticks = ['stdev' for k in range(len(tags))]
    meas = [np.std(all_outs_['low_tot'].values[jj::5]) for jj in range(5)]
    meas2 = [np.std(all_outs_['high_tot'].values[jj::5]) for jj in range(5)]
    lulz = {'ticker':ticks, 'strat':tags, 'low_tot':meas, 'high_tot':meas2}
    lulz = pd.DataFrame.from_dict(lulz)
    all_outs.append(lulz)

    all_outs_ = pd.concat(all_outs, axis = 0)
    all_outs_.to_csv(hdir2 / 'strategy_costs.csv')
