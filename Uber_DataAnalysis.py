# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:25:30 2018

@author: Ibitoye
"""

# This exercise aims to explore skills around time series analysis and visualisation in python
# The data being explored was obtained from Kaggle. 
# The data set contains over 4.5 million Uber pickups in New York City in 2014, 
# This exercise explores the application of few datetime series algorithims like ARIMA, etc.


# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns

import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium 
from folium.plugins import HeatMap

# ingest data by loading CSV into memory. 
# data source = https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city

def uber_2014():
    april14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-apr14.csv', header=0)
    may14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-may14.csv', header=0)
    june14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-jun14.csv', header=0)
    july14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-jul14.csv', header=0)
    august14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-aug14.csv', header=0)
    sep14 = pd.read_csv('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\uberData\\uber-raw-data-sep14.csv', header=0)

    merged_2014 = april14.append([may14, june14, july14, august14, sep14], ignore_index=True)
    return merged_2014

uberData = uber_2014()

uberData.sample(13)


# Explorative Data Analysis and Visualisation

uberData.info() # uberData[Date/Time] needs to be pd.to_datetime format. Not object

# Convert uberData['Date/Time'] to time 
uberData['Date/Time'] = uberData['Date/Time'].map(pd.to_datetime)
uberData.sample(15)

# make uberData['Date/Time'] index [0]
uberDataT = uberData
uberDataT.set_index(['Date/Time'], inplace = False)
uberDataT.sample(15)

# check for missing data 
uberDataT.isnull().sample(15)
uberDataT.isnull().info()
uberDataT.isnull().describe()

# Extract hour, day and month from timeseries data
uberDataT.head(3)

def get_dom(dt):
    return dt.day
uberDataT['Day Of Month'] = uberDataT['Date/Time'].map(get_dom)

def get_weekday(dt):
    return dt.dayofweek
uberDataT['WeekDay'] = uberDataT['Date/Time'].map(get_weekday)

def get_hour(dt):
    return dt.hour
uberDataT['Hour'] = uberDataT['Date/Time'].map(get_hour)

def get_month(dt):
    return dt.month
uberDataT['Month'] = uberDataT['Date/Time'].map(get_month)

uberDataT.sample(15)

# Set index to date time in readiness for time series analysis.
uberDataT = uberDataT.set_index('Date/Time', inplace=False)
uberDataT.sample(15)

# Data Exploration.

# what is the outlook on pick up rate per hour
day_rate = uberDataT.groupby('Hour')
dayRate = day_rate.sum()

dayRate_perHour = dayRate[['WeekDay', 'Month']]
dayRate_perHour.head()

dayRate_perHour.plot(kind='bar',figsize=(10,5))
plt.title('Pick up rate') 

# Turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='green')
plt.show()

# which hour of the day has the most pick up rate
checkcount = uberDataT['Hour'].value_counts()
checkcount

checkcount.plot(kind='bar',figsize=(10,5), color='blue', alpha=0.85)
# Turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')

# what is the outlook of pick up rate the data presents
byDate = pd.crosstab(uberDataT['Day Of Month'],uberDataT['Hour'])
byDate
plt.figure(figsize= (15,10))
sns.heatmap(byDate, linewidth=.5, cmap='coolwarm', annot=False)


# FEATURE ENGINEERING - using domain knowledge of the data to create features that make machine learning algorithms work.

# Exploring data for cleansing and wrangling. 
uberData.info()
uberData.describe()

# compare pickup rates of different months

# Feature Engineering 2 - object oriented method of extracting useful columns from original data.
def create_day_series(uberData):
    
    # Grouping by Date/Time to calculate number of trips
    day_uberData = pd.Series(uberDataT.groupby(['Date/Time']).size())
    
    # setting Date/Time as index
    day_uberData.index = pd.DatetimeIndex(day_uberData.index)
    
    # Resampling to daily trips
    day_uberData = day_uberData.resample('1D').apply(np.sum)
    
    return day_uberData

day_uber2014 = create_day_series(uberData)
day_uber2014.head()

plt.plot(day_uber2014)


# Visualise pick up rate via geographic representation
# ingest shape file for new york
nymap = gpd.read_file('C:\\Users\\Ibitoye\\OneDrive\\edX_DATA_SCIENCE\\miniProject\\NY_shapelyFILE\\geo_export_f406be96-9d42-4c64-afca-7f1a38b5d2d0.shp')

# plot shape file
fig, ax = plt.subplots(figsize=(15,15))
nymap.plot(ax=ax)

#plot district
distric_map = folium.Map(location=[42.5, -75.5], zoom_start=7, tiles='cartodbpositron' )
print('default map crs: ',distric_map.crs)

# convert it to the projection of our folium openstreetmap
nymap = distric_map.to_crs({'init':'epsg:3857'})

#show plot
folium.GeoJson(nymap).add_to(distric_map)
distric_map


max_amount = float(uberDataT['Month'].max())

hmap = folium.Map(location=[42.5, -75.5], zoom_start=7, )

hm_wide = HeatMap( list(zip(uberDataT.lat.values, uberData.lon.values, uberDataT.Month.values)),
                   min_opacity=0.2,
                   max_val=max_amount,
                   radius=17, blur=15, 
                   max_zoom=1, 
                 )

folium.GeoJson(distric_map).add_to(hmap)
hmap.add_child(hm_wide)


# ARIMA

# Before going any further into the analysis, the series has to be made stationary.
# Stationarity is the property of exhibiting constant statistical properties (mean, variance, autocorrelation, etc.). 
# If the mean of a time-series increases over time, then it’s not stationary.

#Checking trend and autocorrelation
def initial_plots(day_uber2014, num_lag):

    #Original timeseries plot
    plt.figure(1)
    plt.plot(day_uber2014)
    plt.title('Original data across time')
    plt.figure(2)
    #Autocorrelation plot
    plot_acf(day_uber2014, lags = num_lag)
    plt.title('Autocorrelation plot')
    #Partial Autocorrelation plot
    plot_pacf(day_uber2014, lags = num_lag)
    plt.title('Partial autocorrelation plot')
    
    plt.show()

    
#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(day_uber2014)[1]))

#plotting
initial_plots(day_uber2014, 45)


#Defining RMSE
def rmse(x,y):
    return sqrt(mean_squared_error(x,y))

#fitting ARIMA model on dataset
def SARIMAX_call(day_uber2014,p_list,d_list,q_list,P_list,D_list,Q_list,s_list,test_period):    
    
    #Splitting into training and testing
    training_ts = day_uber2014[:-test_period]
    
    testing_ts = day_uber2014[len(day_uber2014)-test_period:]
    
    error_table = pd.DataFrame(columns = ['p','d','q','P','D','Q','s','AIC','BIC','RMSE'],\
                                                           index = range(len(ns_ar)*len(ns_diff)*len(ns_ma)*len(s_ar)\
                                                                         *len(s_diff)*len(s_ma)*len(s_list)))
    count = 0
    
    for p in p_list:
        for d in d_list:
            for q in q_list:
                for P in P_list:
                    for D in D_list:
                        for Q in Q_list:
                            for s in s_list:
                                #fitting the model
                                SARIMAX_model = SARIMAX(training_ts.astype(float),\
                                                        order=(p,d,q),\
                                                        seasonal_order=(P,D,Q,s),\
                                                        enforce_invertibility=False)
                                SARIMAX_model_fit = SARIMAX_model.fit(disp=0)
                                AIC = np.round(SARIMAX_model_fit.aic,2)
                                BIC = np.round(SARIMAX_model_fit.bic,2)
                                predictions = SARIMAX_model_fit.forecast(steps=test_period,typ='levels')
                                RMSE = rmse(testing_ts.values,predictions.values)                                

                                #populating error table
                                error_table['p'][count] = p
                                error_table['d'][count] = d
                                error_table['q'][count] = q
                                error_table['P'][count] = P
                                error_table['D'][count] = D
                                error_table['Q'][count] = Q
                                error_table['s'][count] = s
                                error_table['AIC'][count] = AIC
                                error_table['BIC'][count] = BIC
                                error_table['RMSE'][count] = RMSE
                                
                                count+=1 #incrementing count        
    
    #returning the fitted model and values
    return error_table

ns_ar = [0,1,2]
ns_diff = [1]
ns_ma = [0,1,2]
s_ar = [0,1]
s_diff = [0,1] 
s_ma = [1,2]
s_list = [7]

error_table = SARIMAX_call(day_uber2014,ns_ar,ns_diff,ns_ma,s_ar,s_diff,s_ma,s_list,30)


# printing top 5 lowest RMSE from error table
error_table.sort_values(by='RMSE').head(5)


#storing differenced series
diff_series = day_uber2014.diff(periods=1)

#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))


initial_plots(diff_series.dropna(), 30)



#Predicting values using the fitted model
def predict(day_uber2014,p,d,q,P,D,Q,s,n_days,conf):
    
    #Splitting into training and testing
    training_ts = day_uber2014[:-n_days]
    #testing_ts = day_uber2014[len(day_uber2014)-n_days:]
    
    #fitting the model
    SARIMAX_model = SARIMAX(training_ts.astype(float),\
                            order=(p,d,q),\
                            seasonal_order=(P,D,Q,s),\
                            enforce_invertibility=False)
    SARIMAX_model_fit = SARIMAX_model.fit(disp=0)
    
    #Predicting
    SARIMAX_prediction = pd.DataFrame(SARIMAX_model_fit.forecast(steps=n_days,alpha=(1-conf)).values,\
                          columns=['Prediction'])
    SARIMAX_prediction.index = pd.date_range(training_ts.index.max()+1,periods=n_days)
    #Returning predicitons
    return SARIMAX_prediction

    
    #Plotting
    plt.figure(4)
    plt.title('Plot of original data and predicted values using the ARIMA model')
    plt.xlabel('Time')
    plt.ylabel('Number of Trips')
    plt.plot(day_uber2014[1:],'k-', label='Original data')
    plt.plot(SARIMAX_prediction,'b--', label='Next {}days predicted values'.format(n_days))
    plt.legend()
    plt.show()
    

#Predicting the values and builing an 80% confidence interval
prediction = predict(day_uber2014,0,1,0,0,1,2,7,7,0.80)


















 







def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)
    

training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1))-1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for iteration in range(100):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_outputs - outputs
    
    adjustments = error*sigmoid_derivative(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustments)
    
    print ("Synaptic weights after training")
    print(synaptic_weights) 
    
    print("Outputs after training: ")
    print(outputs)
