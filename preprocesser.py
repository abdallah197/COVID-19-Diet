import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  


def clean_data(path):
    data = pd.read_csv(path)
    data['Rec_Rate'] = data[['Confirmed', 'Active', 'Recovered']].apply(lambda x: (x['Recovered']/(x['Confirmed'] x['Active']))*100, axis=1)
    data['Total Confirmed'] = data[['Confirmed', 'Population']].apply(lambda row: (row['Confirmed']/100)*row['Population'], axis=1)
    data = data.sort_values(by=['Rec_Rate'], ascending=False)

    median = data.Undernourished[data.Undernourished != "<2.5"].median()
    data['Obesity'].interpolate(method='linear', inplace=True, limit_direction="both")
    data = data.loc[data['Total Confirmed']>= 5000]

    data['Undernourished'].replace(to_replace = '<2.5', value = median, inplace = True)
    data['Undernourished'] = data['Undernourished'].apply(lambda x : float(x))
    data['Undernourished'].interpolate(method='linear', inplace=True, limit_direction="both")
    print(data.isna().sum().sum())
    return data