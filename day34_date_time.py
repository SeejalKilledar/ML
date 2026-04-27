import numpy as np
import pandas as pd

date = pd.read_csv('orders.csv')
time = pd.read_csv('messages.csv')

print(date.head())
print(time.head())

print(date.info())
print(time.info())

# date is object, to access the dates we need to convert
date['date'] = pd.to_datetime(date['date'])
print(date.info())

# Extract year
date['date_year'] = date['date'].dt.year

# Extract month
date['date_month'] = date['date'].dt.month

#Extract day
date['date_day'] = date['date'].dt.day

#Extract Quarter
date['date_month_name'] = date['date'].dt.month_name()

#Extract day of week
date['date_dayofweek'] = date['date'].dt.dayofweek

# Extract day of week
date['date_dow_name'] = date['date'].dt.day_name()

# is weekend
date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday', 'Saturday']),1,0)

# extract week of the year
date['date_week'] = date['date'].dt.week

# extract quarter
date['quarter'] = date['date'].dt.quater

# extract semester
date['semester'] = np.where(date['quarter'].isin([1,2]), 1,2)

# extract time elapsed between dates
import datetime
today = datetime.datetime.today()

print(today - date['date'])
print((today - date['date']).dt.days)
print((today-date['date'])/np.timedelta64(1,'M'),0)


print(date)

time['date'] = pd.to_datetime(time['date'])

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second

# extract time
time['time'] = time['date'].dt.time

# time difference
print(today-time['date'])

# in sec
print(today-time['date']/np.timedelta64(1,'s'))

# in minutes
print(today-time['date']/np.timedelta64(1,'m'))


# in hours
print(today-time['date']/np.timedelta64(1,'h'))

# print(time.info())
