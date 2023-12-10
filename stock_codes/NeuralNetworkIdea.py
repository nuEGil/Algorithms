import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
'''
make sure you know how many time points are available in the metadata
you need to fix the h5 file data thing some how because if this thing
cant open that data then it wont train


-- basic model - it's not designed to handle sequential data. this is just a proof of concept

'''

class FinanceDataGen(tf.keras.utils.Sequence):

    # modify this to get the correct data setup for the financial class.
    def __init__(self, args_ ):
        self.meta_data = args_['meta_data']
        self.batch_size = args_['batch_size']
        self.win_hist = args_['win_hist']
        self.sub_batsize = args_['sub_batsize']
        self.cats = args_['cats']
        self.shuffle = True

        self.indices = self.meta_data.index.tolist()
    def __len__(self):
        return int(self.meta_data.shape[0] // self.batch_size)
    
    def __getitem__(self, idx):
        '''
        read in some spread sheets from the thing
        run data org - calls rolling norm
        '''
        X=[]
        Y=[]

        # open up a couple of files and normalize and tho the ting
        for ii in range(idx*self.batch_size, (idx + 1) * self.batch_size):
            dat0 = pd.read_csv(self.meta_data['path'].iloc[ii])
            X_, Y_ = self.data_org(dat0) # this is going to get nasty watch out
            X.append(X_)
            Y.append(Y_)

        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)
        cuts = 1024
        if X.shape[0]>cuts:
            permy = np.random.permutation(cuts)
            X = X[permy,...]
            Y = Y[permy,...]
        # print('DDD'*8)
        # print(X.shape, Y.shape)
        return X, Y

    # section for normalization and what not
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    def rolling_norm(self, x_, taggo, win = 8):
        # rolling normalization of data
        mean_x = x_.rolling(win).mean()
        mean_x = mean_x.fillna(x_.values[0])
        std_x = x_.rolling(win).std()

        std_x = std_x.fillna(1)
        eps = 0.001
        # if taggo == 0:
        X = (x_.values - mean_x.values ) / (std_x.values + eps)
        X[np.isnan(X)] = 0
        return X
    
    def data_org(self, dat_in):
        # DO SOME MAGIC HERE TO ENSURE THAT REGARDLESS THE ORIGINAL LENGTH,
        # THE FILES ALL COME OUT TO LIKE 500 samples or some arbitrary number.

        #initial definition of variables we need, so mprice, and dmprice
        dat_in['mprice'] = (dat_in['Low'] + dat_in['High'] + dat_in['Close'])/3
        dat_in['dmprice']  = dat_in['mprice'].diff()
        dat_in['dvol']  = dat_in['Volume'].diff()
        dat_in = dat_in.iloc[1::,:]

        #fields we want to operate on
        fields = ['mprice', 'dmprice', 'Volume', 'dvol']
        taggs = [0,0,1,1]
        dats = [self.rolling_norm(dat_in[ff], tt, win = 5) for (ff, tt) in zip(fields, taggs)]
        sub_df = pd.DataFrame.from_dict(dict(zip(fields, dats)))

        # this is the normalized data
        sub_df = sub_df.iloc[1:-1, :]
        # print(sub_df.head())

        ys = np.zeros((sub_df.shape[0],))
        cond0 = sub_df['dmprice']>=0
        cond1 = sub_df['dmprice']<0
        ys[cond0] = 1
        ys[cond1] = 0

        XDAT = np.zeros((sub_df.shape[0]-(2*self.win_hist), self.win_hist, len(fields)), dtype = np.float64) # make self.win_hist spots per field
        YDAT = np.zeros((sub_df.shape[0]-(2*self.win_hist),), dtype = np.float64) # just want price

        for ii in range(0,sub_df.shape[0]-(2*self.win_hist)):
            # get some data points
            XDAT[ii, ...]  = np.concatenate([np.expand_dims(sub_df[ff].values[ii : ii + self.win_hist], axis = -1) for fii, ff in enumerate(fields)], axis = -1)
            YDAT[ii] = 0 + ys[ii + self.win_hist]


        YDAT = tf.keras.utils.to_categorical(YDAT, num_classes = self.cats, dtype = 'float64')
        indi = np.random.permutation(XDAT.shape[0])
        XDAT = XDAT[indi, ...]
        YDAT = YDAT[indi, ...]
        return XDAT, YDAT

class FinanceRegressGen(tf.keras.utils.Sequence):
    # modify this to get the correct data setup for the financial class.
    def __init__(self, args_ ):
        self.meta_data = args_['meta_data']
        self.batch_size = args_['batch_size']
        self.shuffle = True
        self.indices = [i for i in range(len(self.meta_data))]
    def __len__(self):
        return int(len(self.meta_data) // self.batch_size)
    def __getitem__(self, idx):
        '''
        read in some spread sheets from the thing
        run data org - calls rolling norm
        '''
        X=[]
        Y=[]

        # open up a couple of files and normalize and tho the ting
        for ii in range(idx*self.batch_size, (idx + 1) * self.batch_size):
            with h5py.File(self.meta_data[ii], 'r') as f:
                x = np.array(f['X'])
                y = np.array(f['Y'])

            # print(x.shape, y.shape)
            X.append(x)
            Y.append(y)
        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)
        cuts = 500
        if X.shape[0]>cuts:
            permy = np.random.permutation(cuts)
            X = X[permy,...]
            Y = Y[permy,...]

        return X, Y
    # section for normalization and what not
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

class ResNET():
    def __init__(self, model_dir, train_meta, test_meta, win = 64, feats = 4, nout = 2):
        print('initialize - RESNET')
        self.model_dir = model_dir # where to save the network
        self.win = win # number of time points to consider
        self.feats = feats # number of features to consider
        self.nout = nout
        self.train_meta = train_meta
        self.test_meta = test_meta

    def SetArchitecture(self):
        def ResBlock(x, Nfilt_in, Nfilt_ = 32, ksize = 4, activate = 'tanh'):
            # lead layer
            x_ = tf.keras.layers.Conv1D(Nfilt_, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x)
            # match input size layer
            x_ = tf.keras.layers.Conv1D(Nfilt_in, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x_)
            # summing block
            x_ = tf.keras.layers.Add()([x, x_])
            # end of residual portion

            # this last convolution is to change the number of filters at the end
            x_ = tf.keras.layers.Conv1D(Nfilt_, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x_)
            # do batch normalization
            x_ = tf.keras.layers.BatchNormalization()(x_)
            return x_
        Nblocks = 8  # number of residual block
        nfilts_ = 16 # number of filters per kernel in the blocks
        ksize_ = 4   # kernel size per filter
        dense_units = [2**p for p in range(5,0,-1)] # number of dense units

        # set the input layer
        a0 = tf.keras.layers.Input((self.win, self.feats))

        # set the residual layers
        for i in range(Nblocks):
            if i == 0:
                a1 = ResBlock(a0, self.feats, ksize = ksize_, Nfilt_ = nfilts_, activate = 'tanh')
            else:
                a1 = ResBlock(a1, nfilts_, ksize = ksize_, Nfilt_ = nfilts_, activate = 'tanh')
            if i%2 == 1:
                a1 = tf.keras.layers.MaxPooling1D()(a1)

        # set the dense layers
        a1 = tf.keras.layers.Flatten()(a1)
        for di,du in enumerate(dense_units):
            a1 = tf.keras.layers.Dense(du, activation = 'tanh')(a1)
            a1 = tf.keras.layers.Dropout(0.4)(a1)
        # set the output
        a1 = tf.keras.layers.Dense(self.nout, activation = 'softmax')(a1)

        model = tf.keras.Model(inputs = a0, outputs = a1)
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['acc'] )
        model.save(self.model_dir / "models" / "ver_0.hdf5")

        self.model = model
    def learn(self):
        train_args = {'meta_data':self.train_meta, 'batch_size':3, 'win_hist':self.win,
                      'sub_batsize':100, 'cats':2}
        train_gen = FinanceDataGen(train_args)

        test_args = {'meta_data':self.test_meta, 'batch_size':3, 'win_hist':self.win,
                      'sub_batsize':100, 'cats':2}
        test_gen = FinanceDataGen(test_args)


        self.model.fit(train_gen, validation_data=test_gen, epochs=100,
                       verbose=1, shuffle=True,
                       callbacks=[tf.keras.callbacks.CSVLogger(self.model_dir / "logs" / "log.csv", separator = ',', append = False),
                                  tf.keras.callbacks.ModelCheckpoint(self.model_dir / "models" / "ver0_{epoch:02d}-{loss:.2f}.hdf5",
                                  save_weights_only=False, save_best_only=False, save_freq = 'epoch')]) # save_freq goes by number of batches
                                  
        print('')

class RegressResNET():
    def __init__(self, model_dir, train_meta, test_meta, win = 50, feats = 8, nout = 50):
        print('initialize - RESNET')
        self.model_dir = model_dir # where to save the network
        self.win = win # number of time points to consider
        self.feats = feats # number of features to consider
        self.nout = nout
        self.train_meta = train_meta
        self.test_meta = test_meta

    def SetArchitecture(self):
        def ResBlock(x, Nfilt_in, Nfilt_ = 32, ksize = 4, activate = 'tanh'):
            # lead layer
            x_ = tf.keras.layers.Conv1D(Nfilt_, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x)
            # match input size layer
            x_ = tf.keras.layers.Conv1D(Nfilt_in, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x_)
            # summing block
            x_ = tf.keras.layers.Add()([x, x_])
            # end of residual portion

            # this last convolution is to change the number of filters at the end
            x_ = tf.keras.layers.Conv1D(Nfilt_, kernel_size = ksize, strides=1, padding='same',
                                        activation=activate, use_bias=True,)(x_)
            # do batch normalization
            x_ = tf.keras.layers.BatchNormalization()(x_)
            return x_

        Nblocks = 8  # number of residual block
        nfilts_ = 16 # number of filters per kernel in the blocks
        ksize_ = 4   # kernel size per filter
        # set the input layer
        a0 = tf.keras.layers.Input((self.win, self.feats))

        # set the residual layers
        for i in range(Nblocks):
            if i == 0:
                a1 = ResBlock(a0, self.feats, ksize = ksize_, Nfilt_ = nfilts_, activate = 'tanh')
            else:
                a1 = ResBlock(a1, nfilts_, ksize = ksize_, Nfilt_ = nfilts_, activate = 'tanh')

        # no max pooling since this one is a regressor model.
        a1 = tf.keras.layers.Conv1D(self.feats, kernel_size = ksize_, strides=1, padding='same',
                                    activation='tanh', use_bias=True,)(a1)

        model = tf.keras.Model(inputs = a0, outputs = a1)
        model.compile(optimizer = "adam", loss = "mse", metrics = ['acc'] )
        model.save(self.model_dir / "models" / "ver_0.hdf5")
        self.model = model

    def learn(self):
        train_args = {'meta_data':self.train_meta, 'batch_size':3,}
        train_gen = FinanceRegressGen(train_args)

        test_args = {'meta_data':self.test_meta, 'batch_size':3,}
        test_gen = FinanceRegressGen(test_args)


        self.model.fit(train_gen, validation_data=test_gen, epochs=100,
                       verbose=1, shuffle=True,
                       callbacks=[tf.keras.callbacks.CSVLogger(str(self.model_dir / "logs" / "log.csv"), separator = ',', append = False),
                                  tf.keras.callbacks.ModelCheckpoint(str(self.model_dir / "models" / "ver0_{epoch:02d}-{loss:.2f}.hdf5"),
                                  save_weights_only=False, save_best_only=False, save_freq = 'epoch')])

def worker():
    
    hdir = Path('/Stocks')
    model_dir = Path('stocks/NETWORKS/ResNet/BIG_BOI_DAT')
    meta_dat_dir = Path('stocks/NETWORKS/MetaDataSplit')

    # check to make sure that the paths exists and we have a starting log file
    if not os.path.exists(model_dir / "logs"):
        os.makedirs(model_dir / "logs")
        with open(model_dir/"logs"/"log.csv", 'w') as fp:
            fp.write("")

    if not os.path.exists(model_dir / "models"):
        os.makedirs(model_dir / "models")

    train_meta_dat = pd.read_csv(meta_dat_dir / "Stocks_train_dat.csv")
    test_meta_dat = pd.read_csv(meta_dat_dir / "Stocks_test_dat.csv")


    CHOLO = ResNET(model_dir, train_meta_dat, test_meta_dat) # instantiate the model
    CHOLO.SetArchitecture() # set the architecture up
    CHOLO.learn() # do the training now

def regress_worker():

    model_dir = Path('stocks/NETWORKS/RegressResNET/ewm-4_IN-50_OUT-50/')
    hdir = Path('stocks/NETWORKS/DATA_SETS/REGRESSOR/ewm-4_IN-50_OUT-50/')

    train_hdir = hdir / "train"
    test_hdir = hdir / "test"


    # check to make sure that the paths exists and we have a starting log file
    if not os.path.exists(model_dir / "logs"):
        os.makedirs(model_dir / "logs")
        with open(model_dir/"logs"/"log.csv", 'w') as fp:
            fp.write("")
    if not os.path.exists(model_dir / "models"):
        os.makedirs(model_dir / "models")


    train_files_ = sorted(list(map(str, train_hdir.glob('*.hdf5'))))
    test_files_ = sorted(list(map(str, test_hdir.glob('*.hdf5'))))

    CHOLO = RegressResNET(model_dir, train_files_, test_files_)
    CHOLO.SetArchitecture() # set the architecture up
    CHOLO.learn() # do the training now

if __name__ =='__main__':
    print('WORKING')
    # worker()
    regress_worker()
