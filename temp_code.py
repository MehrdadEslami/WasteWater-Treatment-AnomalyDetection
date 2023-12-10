"""*****************************************************Pandas**********************"""
import pandas as pd
from datetime import datetime
import numpy as np

"""Reading DataFrame From CSV file"""
dataset = pd.read_csv(filepath_or_buffer='./.data/HitBTC_BTCUSD_1h.csv', parse_dates=[1])

"""saving DataFrame to CSV file"""
dataset.to_csv('./.data/yfinance/BTCUSD_d_20200101_20220818')

"""insert a colum to dataframe"""
dframe.insert(5, 'class',value=label_col)
"""convert dataframe """
temp_dataframe.insert(len(temp_dataframe.columns), 'class', value=label)

"""Creating a DataFrame from dict, 'col1' , 'col2' is columns and values are value index start 0 """
e = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data = e)

"""change dtype when creating"""
df = pd.DataFrame(data=d, dtype=np.int8)

"""Constructing DataFrame from numpy ndarray: if ignore columns and index pandas create by itself started from 0"""
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'], index=['m','k','j'])

"""Constructing DataFrame from dataclass:"""
from dataclasses import make_dataclass
Point = make_dataclass("Point", [("x", int), ("y", int)])
pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])


"""Retriving DataFrame Columns"""
dataset = dataset[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume BTC']]

"""Convert a Date Columns to datetime format. also applying a function to a columns"""
# dataset['Date'] = dataset['Date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d  %H-%p'))

"""retriving a rows ( index)"""
dataset2 = dataset.loc[0:5]

print('x_dataset[0]',x_dataset[0]) # row 0 in dataframe
print('x_dataset[0][2]',x_dataset[0][2]) # row 0 , col 2 in dataframe
print('x_dataset[1][2]',x_dataset[1][2]) # row 1 , col 2 in dataframe
print('x_dataset[2][2]',x_dataset[2][2])# row 2 , col 2 in dataframe

print('x_dataset[:,2]',x_dataset[:,2]) # all row , col 2 in dataframe
print(dataset2)
#
# print('dataset2 coloms',dataset2.columns)
# print('dataset2 shape',dataset2.shape)
# print('dataset2 type',dataset2.dtype)



""""OrderDic"""
from collections import OrderedDict

print("\nThis is an Ordered Dict:\n")
od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3
od['d'] = 4

for key, value in od.items():
    print(key, value)


"""Downloading Data From yFinance"""
import yfinance as yf
data = yf.download('BTC-USD', start='2020-01-01', end='2022-08-18')
df = data[["Open", "High", "Low", "Close", "Volume"]]
df = pd.DataFrame(data=df)
df.to_csv('./.data/yfinance/BTCUSD_d_20200101_20220818')



"""********************************************************NUMPY*****************************"""

np_array = np.array([1,2,3,4,5,6])
"""convert numpy array dtype"""
np_array = np_array.astype(int)
"""Random Number"""
np.random.rand(5)  # generate 5 number between 0,1
x = int(np.random.rand(5)*10 + 4)  # generate 5 number between 4 , 13
random.randrange(1,4) # betwwen 1,4


"""********************************************************OS*****************************"""
print("Create Folder is started")
try:
    os.chdir(base_dir)
except (FileNotFoundError):
    os.mkdir(base_dir)


os.chdir(original_dataset_dir)
classes = os.listdir()
for i in classes:
    os.mkdir(train_dir + '/' + i)
    os.mkdir(validation_dir + '/' + i)

    os.chdir(original_dataset_dir + '/' + i)
    data = os.listdir()

    data_count = data.__len__()
    fnames_train = random.sample(data, int(percent['train'] * data_count / 100))
    print("copy Train file to folder" + i + "started")
    for fname in fnames_train:
        src = os.path.join(original_dataset_dir + '/' + i, fname)
        dst = os.path.join(train_dir + '/' + i, fname)
        shutil.copyfile(src, dst)
