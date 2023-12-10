import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
'''
goes from 89 to  2020

make sure you are able to open the meta data and the daily Data
read dates from both, then align a vector of all the features including
prices


got the excell sheets alligned just need
next write code to open daily and monthly info by ticker name, and
create time vector.

make this dump the new sheet to some folder so we dont have to put this vector together every time .
'''
def make_date_dicts():
    doub_year = [str(i) for i in range(50,100)]
    doub_year.extend(['{:02}'.format(i) for i in range(21)])

    year = ['{}'.format(ii) for ii in range(1950,2021)]
    year_dict = dict(zip(doub_year, year))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep','Oct','Nov', 'Dec']
    nmonth = ['{:02}'.format(i+1) for i in range(12)]

    month_dict = dict(zip(months,nmonth))
    # print(month_dict)
    # print(year_dict)
    return [month_dict, year_dict]
def align_day_month_names():
    #Get the daily data
    out_dir = Path('stocks')
    daily_dirs = ['stocks/4538_7213_bundle_archive/Data/Stocks',
                  'stocks/4538_7213_bundle_archive/Data/ETFs']
    
    quarter_dirs = ['stocks/guru_focus/S&P500-20200630T010120Z-001/SP500_solo/quarterly',]
    def get_the_beepers(dirs_, extension = '*.csv'):
        # Get the quarterly data
        paths_ = [Path(dir_) / '' for dir_ in dirs_]
        files_ = []
        for qp in paths_:
            files_.extend(list(map(str, qp.glob(extension))))
        files_= sorted(files_)
        return files_

    daily_files = get_the_beepers(daily_dirs,'*.us.txt')
    quarter_files = get_the_beepers(quarter_dirs,'*.csv')

    day_tags = [dd.split('\\')[-1].strip('.us.txt') for dd in daily_files]
    quarter_tags = [qd.split('\\')[-1].strip('.csv').lower() for qd in quarter_files]

    # print(day_tags)
    # print(quarter_tags)
    pd_day_tag = pd.DataFrame(np.array([day_tags, daily_files]).T, columns = ['tag','day_names'])
    pd_quarter_tag = pd.DataFrame(np.array([quarter_tags, quarter_files]).T, columns = ['tag','quarter_names'])

    pd_complete = pd_quarter_tag.merge(pd_day_tag)#, how = 'inner', left_on='tag')
    # print(pd_complete.shape)
    cond = pd_complete['day_names'].str.split(".").str[-3].str.split('\\').str[-1].str.lower()
    cond2 = pd_complete['quarter_names'].str.split(".").str[-2].str.split('\\').str[-1].str.lower()
    # hhh = cond[61] == cond2[61]
    # print(hhh)
    # print(cond[61])
    # print(cond2[61])
    pd_complete = pd_complete[cond == cond2]
    pd_complete.to_csv(out_dir/'data_by_ticker.csv')

def upsample_quarterly():
    # Read the alignment file
    meta_dat = pd.read_csv('stocks/data_by_ticker.csv')
    out_dir =  Path('stocks/4538_7213_bundle_archive/SP500_wgurufeats/')
    # print(meta_dat.head())

    n_files = meta_dat.shape[0]
    for mi_n in range(n_files):
        qq = pd.read_csv(meta_dat['quarter_names'][mi_n], header = None)
        dd = pd.read_csv(meta_dat['day_names'][mi_n])

        print('working:{}'.format(meta_dat['tag'][mi_n]))

        qq.iloc[0,2::] = qq.iloc[0,2::].apply(lambda x: x.replace(x[0:3], month_dict[x[0:3]]+'-' ))
        def swap(x):
            tmp_ = x.split('-')
            tmp_ = tmp_[::-1]
            tmp_.extend(['01'])
            return '-'.join(tmp_)
        qq.iloc[0,2::] = qq.iloc[0,2::].apply(swap)


        # Add fields to the data frame so that we can get them here in a second
        feats = qq.shape[0]

        ki_o = 0
        for fi in range(feats):
            # name repeats sometimes fucking apparently.
            # print(qq.iloc[fi,0])
            tmp_name = qq.iloc[fi,1]
            if tmp_name in dd.columns:
                tmp_name = tmp_name+'new_{}'.format(ki_o)
                ki_o+=1
            dd[tmp_name] = 0

        for dai in range(2,qq.shape[1]-3):
            cond0 = pd.to_datetime(dd['Date']) >= pd.to_datetime(qq.iloc[0,dai])
            cond1 = pd.to_datetime(dd['Date']) <= pd.to_datetime(qq.iloc[0,dai+1] ) + datetime.timedelta(days = 7)
            cond = cond0 & cond1
            if sum(cond)==0:
                pass
            else:
                # print(cond[cond].index)
                ho = cond[cond].index[0]
                hi = cond[cond].index[-1]
                dd.iloc[ho:hi,7::] = np.tile(qq.iloc[:, dai].T, (hi-ho,1))

        dd.to_csv(out_dir/'{}.csv'.format(meta_dat['tag'][mi_n]))

if __name__ == '__main__':
    # Make dictionaries for month and year alignment
    month_dict, year_dict = make_date_dicts()

    if not Path('stocks/data_by_ticker.csv').exists():
        print('the alignment file does not exist. creating it now')
        align_day_month_names()
    else:
        print('alignment file exists reading it now ')
        upsample_quarterly()
