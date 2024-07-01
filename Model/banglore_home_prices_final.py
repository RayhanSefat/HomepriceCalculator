import os
default_directory = "F:\Homeprice Calculator\Model"
os.chdir(default_directory)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

def is_float(x):
    try:
        float(x)
    except:
        return False
    
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   

df1 = pd.read_csv("bengaluru_house_prices.csv")
print(df1.head())
print(df1['area_type'].value_counts())

df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.shape)
print(df2.isnull().sum())

# as the number of null values is very less than the size of the entire dataset
# we are simply throwing them away from our considerable scope
df3 = df2.dropna()
print(df3.isnull().sum())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3['bhk'].unique())

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())
print(df4.isnull().sum())

df4 = df4.dropna()
print(df4.isnull().sum())