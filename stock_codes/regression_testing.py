import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn import svm
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
Want to make a predictive model
Stanadardization
1. compute the longest mean possible
2. compute the longest stdev
3. xi = (xi - mui) / (stdevi) - need to store these numbers

Input - windows N long of
1. previous prices
2. previous volumes
3. previous dprice
4. previous dvol

Output - windows N long of
1. price
2. volume

As a note, this probably wont work all that well right off the bat
need to do short term standardization!

Think on some descriptors or something to mess with this classificatino and prediction
pip install tensorflow to get the neural network stuff going on this.
'''
class learner():
    '''
    all this does is list the methods we expect in a learner function
    '''
    def __init__(self, x, hdir):
        print('initialize')
        self.x = x
        self.hdir = hdir

    def backtest(self,buy_strat, low, high, tag):
        print('backtesting')
        #figure out the cash flow .
        money_low = -1*buy_strat*low
        money_high = -1*buy_strat*high
        print('strategy:{}'.format(tag))
        print('low total', sum(money_low))
        print('high total', sum(money_high))
        print('\n')
        return [sum(money_low), sum(money_high)]

class LinRegLearner(learner):
    def __init__(self, x, hdir, win =5):
        '''
        this one tries to do a linear regression. kinda works but not really
        '''
        print('initialize - LinRegLearner')
        self.x = x
        self.win = 0 + win
        self.hdir = hdir

    def data_org(self):
        print('LINREG LEARNER DATA ORG')
        self.x['mprice'] = (self.x['Low'] + self.x['High'] + self.x['Close'])/3
        self.x['dmprice']  = self.x['mprice'].diff()
        self.x['dvol']  = self.x['Volume'].diff()
        self.x = self.x.iloc[1::, :]
        # get what we need to standardize
        fields = ['mprice', 'dmprice', 'Volume', 'dvol']
        mprice_mean = [np.mean(self.x[fi]) for fi in fields]
        mprice_stdv = [np.std(self.x[fi]) for fi in fields]


        XDAT = np.zeros((self.x.shape[0]-(2*self.win), self.win*len(fields))) # make self.win spots per field
        YDAT = np.zeros((self.x.shape[0]-(2*self.win), self.win*len(fields))) # just want price

        for ii in range(0,self.x.shape[0]-(2*self.win)):
            # get some data points
            cloro = np.concatenate([((self.x[ff].values[ii : ii+self.win] - mprice_mean[fii])) / mprice_stdv[fii] for fii, ff in enumerate(fields)])
            # get some future data points
            cloro2 = np.concatenate([((self.x[ff].values[ii+self.win : ii+(2*self.win)] - mprice_mean[fii])) / mprice_stdv[fii] for fii, ff in enumerate(fields)])
            # print(cloro.shape)
            # update entries in the self.x data set and Y data set
            XDAT[ii, :] = 0 + cloro
            YDAT[ii, :] = 0 + cloro2

        # save the XDAT and the YDAT
        self.XDAT = 0 + XDAT
        self.YDAT = 0 + YDAT
    
    def learn(self):
        print('LINREG LEARNER LEARN')
        # perform the linear regression
        reg = LinearRegression().fit(self.XDAT, self.YDAT)
        self.YPRED = reg.predict(self.XDAT)
        print(self.YPRED.shape)
        # print stuff. save coefficients
        np.savetxt(self.hdir / 'linreg_coefficients.csv', reg.coef_, delimiter = ',')
    
    def plotter(self):
        print('LINREG LEARNER PLOT')
        plt.figure()
        for ii in range(20):
            plt.subplot(5,4,ii+1)
            plt.scatter(self.YPRED[:,ii], self.YDAT[:,ii])
            _dodo = np.concatenate([self.YPRED[:,ii], self.YDAT[:,ii]])
            plt.xlim([np.min(_dodo), np.max(_dodo)])
            plt.ylim([np.min(_dodo), np.max(_dodo)])

        plt.show()

class UpDownSVC(learner):
    def __init__(self, x, hdir, win = 50):
        print('initialize')
        self.x = x
        self.hdir = hdir
        self.win = win

    def rolling_norm(self, x_, taggo, win = 5):
        # rolling normalization
        mean_x = x_.rolling(win).mean()
        mean_x = mean_x.fillna(x_.values[0])
        std_x = x_.rolling(win).std()
        std_x = std_x.fillna(1)
        if taggo == 0:
            X = (x_.values - mean_x.values) / std_x.values
        else:
            X = np.log10(x_.values - mean_x.values) - np.log10(std_x.values)
        X[np.isnan(X)] = 0
        return X

    def data_org(self):
        print('UPDOWN SVC')
        # print(self.x.columns)
        #initial definition of variables we need, so mprice, and dmprice
        self.x['mprice'] = (self.x['Low'] + self.x['High'] + self.x['Close'])/3
        self.x['dmprice']  = self.x['mprice'].diff()
        self.x['dvol']  = self.x['Volume'].diff()
        self.x = self.x.iloc[1::,:]

        #fields we want to operate on
        fields = ['mprice', 'dmprice', 'Volume', 'dvol']
        taggs = [0,0,1,1]

        print(self.x.describe())
        dats = [self.rolling_norm(self.x[ff], tt, win = 50) for (ff, tt) in zip(fields, taggs)]

        sub_df = pd.DataFrame.from_dict(dict(zip(fields,dats)))
        print(sub_df.head())
        self.x.update(sub_df)
        self.x = self.x.iloc[1:-1, :]
        # print(sub_df.head())

        ys = np.zeros((self.x.shape[0],))
        cond0 = self.x['dmprice']>=0
        ys[cond0] = 1

        XDAT = np.zeros((self.x.shape[0]-(2*self.win), self.win*len(fields))) # make self.win spots per field
        YDAT = np.zeros((self.x.shape[0]-(2*self.win), )) # just want price
        #
        for ii in range(0,self.x.shape[0]-(2*self.win)):
            # get some data points
            cloro = np.concatenate([self.x[ff].values[ii : ii + self.win] for fii, ff in enumerate(fields)])
            # get some future data points

            # print(cloro.shape)
            # update entries in the self.x data set and Y data set
            XDAT[ii, :] = 0 + cloro
            YDAT[ii,] = 0 + ys[ii + self.win]
        # shuffle the data
        # training test split
        train_perc = 0.9
        train_cut  = int(train_perc * XDAT.shape[0])
        shuff = np.random.permutation(train_cut)

        # shuffle and mix into training and testing sets
        self.train_XDAT = 0 + XDAT[shuff,:]
        self.test_XDAT = 0 + XDAT[train_cut::, :]

        self.train_YDAT = 0 + YDAT[shuff, ]
        self.test_YDAT = 0 + YDAT[train_cut::, ]

        print('shapes')
        print(self.train_XDAT.shape, self.train_YDAT.shape)
        print(self.test_XDAT.shape, self.test_YDAT.shape)

    def learn(self, ticker):
        kernel_ = 'rbf'
        clf = svm.SVC(kernel = kernel_)
        # train on the training data
        clf.fit(self.train_XDAT, self.train_YDAT)
        # test on the testing data
        self.YPRED = clf.predict(self.test_XDAT)
        tn, fp, fn, tp = metrics.confusion_matrix(self.test_YDAT, self.YPRED).ravel()
        tpr = tp / np.sum(self.test_YDAT)
        fpr = fp / np.sum((1-self.test_YDAT))
        # same thing
        # tpr = np.sum((self.YPRED * self.test_YDAT)) / np.sum(self.test_YDAT)
        # fpr = np.sum((self.YPRED * (1 - self.test_YDAT))) / np.sum((1 - self.test_YDAT))
        print('tpr:{}\nfpr:{}'.format(tpr,fpr))

        pickle_out = open(self.hdir / 'svc_{}_{}_tpr-{:.2f}_fpr-{:.2f}.pickle'.format(kernel_, ticker, tpr, fpr),"wb")
        pickle.dump(clf, pickle_out)
        pickle_out.close()

    def ensemble_learn(self):
        kernels_ = ['rbf','linear', 'sigmoid']
        weights = [0.75, 0.125, 0.125]
        clfs = [svm.SVC(kernel = ks_) for ks_ in kernels_]
        clfs = [clf.fit(self.train_XDAT, self.train_YDAT) for clf in clfs]

        # train on the training data
        # test on the testing data
        YPREDS = np.concatenate([np.expand_dims(ww*clf.predict(self.test_XDAT),axis =-1) for (clf,ww) in zip(clfs, weights)],axis = -1)
        YPREDS = np.mean(YPREDS, axis = -1 )
        fpr, tpr, threshes = metrics.roc_curve(self.test_YDAT, YPREDS, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr,tpr)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve AUC:{}'.format(auc))
        plt.show()

def worker():
    np.random.seed(69420)
    
    hdir = Path('google_finance_data')
    model_dir = Path('4538_7213_bundle_archive/models')
    files_ = sorted(list(map(str,hdir.glob('*.csv'))))

    all_outs = []
    for ji, ff in enumerate(files_):
        ticky = ff.split('\\')[-1].strip('.csv')
        if not ticky == 'tsla':
            pass
        else:
            # now we know its gonna be nvidia
            dat_ = pd.read_csv(ff)
            # CHOLO = LinRegLearner(dat_, model_dir, win = 5)
            CHOLO = UpDownSVC(dat_, model_dir)
            # run the cholo
            CHOLO.data_org()
            CHOLO.learn(ticky)
            # CHOLO.ensemble_learn()
            CHOLO.plotter()

if __name__ == '__main__':
    worker()
