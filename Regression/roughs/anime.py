import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani


url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
data = pd.read_csv(url, delimiter=',', header = 'infer')

df_interest = data.loc[data['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany']) & data['Province/State'].isna()]
df_interest.rename(index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
df1 = df_interest.transpose()
df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
df1 = df1.loc[(df1 != 0).any(1)]
df1.index = pd.to_datetime(df1.index)

def makeanimation(i=int):
  plt.legend(df1.columns)
  p = plt.plot(df1[:i].index, df1[:i].values)
  for i in range(0,4):
    p[i].set_color(color[i])

color = ['red', 'green', 'blue', 'orange']
fig = plt.figure(figsize=(8,8))

# configuring the type of the plot

plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom = 0.2, top = 0.9)
plt.ylabel('No of Deaths')
plt.xlabel('Dates')

import matplotlib.animation as ani
animator = ani.FuncAnimation(fig, makeanimation, interval = 100)
plt.show()