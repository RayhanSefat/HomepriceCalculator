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
    
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
    
pd.set_option('display.max_columns', None)

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

df5 = df4.copy()
df5['price_per_sqft'] = (df5['price'] * 100000) / df5['total_sqft']
print(df5.head())

df5.location = df5.location.apply(lambda x: x.strip())
print(df5.head())
location_stats = df5['location'].value_counts(ascending=False)
print(location_stats)
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

print(df5[(df5.total_sqft / df5.bhk) < 300].head())
print(df5.shape)

df6 = df5[-((df5.total_sqft / df5.bhk) < 300)]
print(df6.shape)
print(df6.price_per_sqft.describe())

df7 = remove_pps_outliers(df6)
print(df7.shape)
    
# plot_scatter_chart(df7,"Hebbal")

df8 = remove_bhk_outliers(df7)
print(df8.shape)

# plot_scatter_chart(df8,"Rajaji Nagar")

# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")
# plt.show()

print(df8.bath.unique())
print(df8[df8.bath > 10])

# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("Number of bathrooms")
# plt.ylabel("Count")
# plt.show()

print(df8[df8.bath > df8.bhk + 2])

df9 = df8[df8.bath < df8.bhk + 2]
print(df9.shape)

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head())