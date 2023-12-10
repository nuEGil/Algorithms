import os
import h5py
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
'''
Plan is to just get enough data to make some nice plots and what not
You need a day variable too. M T W TH F because activity

Ok. so new plan. lets just make a regression network that takes
the current 50 days, and generates the next 50 days......
dake date and time into account.....

What if I take a network and train it to take 500 days of features and
dump 500 days of dprice.....

.ewm(com = 50).mean() - is exponential decay moving average


Write code to make training data sets ... need to take 50 day chunks for all signals

# the read opperation on h5py is
with h5py.File('test_read.hdf5', 'r') as f:
    d1 = f['array_1']
    d2 = f['array_2']

    data = d2[d1[:]>1]
'''
def set_data_splits(hdirs):

    files_ = sorted(list(map(str,hdirs.glob('*.us.txt'))))
    ticky_ = [ff.split('\\')[-1].strip('.us.txt') for ff in files_]

    keep_files = []
    lens = []
    keep_ticky = []

    for fi,ff in enumerate(files_):
        try:
            lens_ = pd.read_csv(ff).shape[0]
            keep_ticky.append(ticky_[fi])
            lens.append(lens_)
            keep_files.append(ff)
        except:
            print('something was up with the file man - skipping ')
            print('xxxx', ff)



    metadat = pd.DataFrame.from_dict({'ticker':keep_ticky, 'path':keep_files, 'lens':lens})
    metadat = metadat[metadat['lens']>500] # keep data only with more than 500 time points

    metadat.to_csv(Path('network_dir/') / 'MetaDataSplit' / 'Stocks_all_dat.csv')
    metadat = metadat.sample(frac=1).reset_index(drop=True)

    train_perc = 0.85
    mshs = metadat.shape[0]
    train_dat = metadat.iloc[0:int(train_perc * mshs), :]
    test_dat = metadat.iloc[int(train_perc * mshs)::, :]

    train_dat.to_csv(Path('network_dir/') / 'MetaDataSplit' / 'Stocks_train_dat.csv')
    test_dat.to_csv(Path('network_dir/') / 'MetaDataSplit' / 'Stocks_test_dat.csv')

class FeatureSnipe():
    def __init__(self, meta_dat):
        print('not sure what I want here')
        self.mdat = meta_dat # so an excel sheet that points to data
        self.win = 4 # windo on the ewm filter
        self.observe_Xwin = 50
        self.observe_Ywin = 50

    def cleanup(self):
        del self.XDAT
        del self.YDAT
        del self.DAT
        del self.pass_flag
    
    def whole_norm(self, x, tagg):
        if tagg == 1:
            X = x.ewm(com = self.win).mean()
            X = X - X.mean()
            X = X.values / np.abs(X.values).max()
        else:
            X = x - x.mean()
            X = X.values / np.abs(X.values).max()

        return X
    
    def read(self,itt):
        print('coll')
        self.itt = itt
        data_ = pd.read_csv(self.mdat['path'].iloc[itt]) # so just get the first bit of data
        # Callendar stuff Y-M-D
        data_['Year'] = data_['Date'].str.split('-').str[0].values.astype(int)
        data_['Month'] = data_['Date'].str.split('-').str[1].values.astype(int)
        data_['Day'] = data_['Date'].str.split('-').str[2].values.astype(int)
        # year is going to keep going up so lets check on 7 year cycles
        data_['mod_Year'] = data_['Year'].apply(lambda x: (x % 7)) # rule of 7 lol

        # get the week day too because of autistic and retard buying schedules
        data_['nDay'] = np.array([dt.datetime(yy, mm, dd).weekday() for (yy,mm,dd) in zip(data_['Year'].values, data_['Month'].values, data_['Day'].values)])

        # get the mean price - (1/3) * Low + High + close
        data_['mprice'] = (data_['Low'] + data_['High'] + data_['Close'])/3 # get mprice
        data_['dmprice'] = data_['mprice'].diff() / data_['nDay'].diff()


        # volume is always huge  - so lets log scale it
        eps = 0.000001
        data_['LogVol'] = np.log10(data_['Volume'].values + eps)
        data_['dmprice_dvol'] =  data_['mprice'].diff() / data_['LogVol'].diff()

        tagg = [1,1,1,1,0,0,0,0]
        fields = ['mprice', 'dmprice','LogVol' ,'Volume', 'nDay','Day', 'Month', 'mod_Year']
        for (ff, tg) in zip(fields, tagg):
            data_[ff].iloc[1::] = self.whole_norm(data_[ff].iloc[1::], tg)

        self.DAT = data_[fields].iloc[self.win::]

    def create_dataset(self, dump_dir0, set_0):
        xdat = []
        ydat = []
        for ii in range(self.DAT.shape[0] - (self.observe_Xwin + self.observe_Ywin)):
            xdat.append(np.expand_dims(self.DAT.iloc[ii:ii+self.observe_Xwin].values, axis = 0))
            ydat.append(np.expand_dims(self.DAT.iloc[ii+self.observe_Xwin : ii + self.observe_Xwin + self.observe_Ywin].values, axis = 0))
        self.XDAT = np.concatenate(xdat, axis = 0)
        self.YDAT = np.concatenate(ydat, axis = 0)
        print(self.XDAT.shape, self.YDAT.shape)
        print(np.sum(np.isnan(self.XDAT)), np.sum(np.isnan(self.YDAT)))
        if np.sum(np.isnan(self.XDAT))>0 or np.sum(np.isnan(self.YDAT)) > 0:
            self.pass_flag = 1
        else:
            self.pass_flag = 0

        print(type(self.XDAT))
        print(self.mdat.ticker.iloc[self.itt])
        print(self.XDAT.min(), self.XDAT.max(), self.YDAT.min(), self.YDAT.max(), )
        if not self.pass_flag:
            with h5py.File(dump_dir0 / set_0 / '{}.hdf5'.format(self.mdat.ticker.iloc[self.itt]),'w') as hff:
                hff.create_dataset("X", data = self.XDAT.astype(np.float64))
                hff.create_dataset("Y", data = self.YDAT.astype(np.float64))

    def PlotNChart(self):
        print('plot the features')
        plt.figure()
        plt.subplot(5,1,1)
        plt.plot(self.DAT['mprice'])

        # plt.xlim([0,500])
        plt.subplot(5,1,2)
        plt.plot(self.DAT['dmprice'])
        plt.plot(np.zeros(self.DAT['mprice'].shape))
        # plt.xlim([0,500])
        plt.subplot(5,1,3)
        plt.plot(self.DAT['LogVol'])
        plt.subplot(5,1,4)
        plt.plot(self.DAT['Volume'])

        plt.figure()
        plt.plot(self.XDAT[:,0,0])

        plt.plot(self.YDAT[:,0,0])


        plt.show()

    def check_dataset(self):
        hdir = Path('network_dir/DATA_SETS/REGRESSOR/ewm-4_IN-50_OUT-50')
        train_hdir = hdir / "train"
        test_hdir = hdir / "test"

        train_files_ = sorted(list(map(str, train_hdir.glob('*.hdf5'))))
        test_files_ = sorted(list(map(str, test_hdir.glob('*.hdf5'))))
        files_ = [train_files_, test_files_]

        for fifi in files_:
            for jj in range(len(fifi)):
                print(fifi[jj])

                del_flag = 0 # delete the file if this is a 1.
                with h5py.File(fifi[jj], 'r') as f:
                    print(f.keys())
                    print('X' in f.keys() , 'Y' in f.keys())
                    if 'X' in f.keys() and'Y' in f.keys():
                        x = np.array(f["X"])
                        y = np.array(f["Y"])
                        print(x.shape, y.shape)

                    else:
                        print('deleting'*3, '----'*8)
                        print(fifi[jj])
                        del_flag = 1

                if del_flag == 1:
                    os.remove(fifi[jj])

def worker():
    meta_dat_dir = Path('network_dir/MetaDataSplit')
    dump_dir = Path('network_dir/DATA_SETS/REGRESSOR/ewm-4_IN-50_OUT-50')

    # set_data_splits(hdir) # only need to run this when you dont have training and testing splits
    sets = [ 'test','train']

    for set_  in sets:
        if not os.path.exists(dump_dir / set_):
            os.makedirs(dump_dir / set_)
        data_file = 'Stocks_{}_dat.csv'.format(set_)

        mdat = pd.read_csv(meta_dat_dir / data_file)
        g = FeatureSnipe(mdat)
        for ii in range(mdat.shape[0]):
            try:
                g.read(ii)
                g.create_dataset(dump_dir, set_)
                g.cleanup()
            except:
                print('couldnt do')
                print(mdat.ticker.iloc[ii])

def tester():
    CHOLO = FeatureSnipe('')
    CHOLO.check_dataset()

if __name__ == '__main__':
    # worker()
    tester()
