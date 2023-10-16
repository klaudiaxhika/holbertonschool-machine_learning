#!/usr/bin/env python3
"""
Complete the following script to fill in the missing data points in the pd.DataFrame:

The column Weighted_Price should be removed
missing values in Close should be set to the previous row value
missing values in High, Low, Open should be set to the same row’s Close value
missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop('Weighted_Price', axis=1, inplace=True)
df['Close'].fillna(method='ffill', inplace=True)
df["Volume_(BTC)"].fillna(value=0, inplace=True)
df["Volume_(Currency)"].fillna(value=0, inplace=True)

df = df.fillna({'Open': df['Close'].shift(1, fill_value=0),
                'High': df['Close'].shift(1, fill_value=0),
                'Low': df['Close'].shift(1, fill_value=0)})

print(df.head())
print(df.tail())
