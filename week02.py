# -*- coding: utf-8 -*-
"""
Spyder Editor
print("Hello, World!")
This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dtype_dict ={'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train_data = pd.read_csv("C:\Users\N_Solgi\Desktop\quiz\kc_house_train_data.csv")
test_data=pd.read_csv("C:\Users\N_Solgi\Desktop\quiz\kc_house_test_data.csv")

#2
train_data['bedroom_squared']=train_data['bedrooms']*train_data['bedrooms']
train_data['bed_bath_rooms']=train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living']=np.log10(train_data['sqft_living'])
train_data['lat_plus_long']=train_data['lat']+train_data['long']

#3
test_data['bedroom_squared']=test_data['bedrooms']*test_data['bedrooms']
test_data['bed_bath_rooms']=test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living']=np.log10(test_data['sqft_living'])
test_data['lat_plus_long']=test_data['lat']+test_data['long']

#4
f1=plt.figure(1)
plt.scatter(train_data['sqft_living'],train_data['price'], color='blue')
f1.show()

f2=plt.figure(2)
plt.scatter(train_data['log_sqft_living'],train_data['price'],color='red')
f2.show()


#5
features_of_model_1=['sqft_living', 'bedrooms','bathrooms','lat','long']
features_of_model_2=['sqft_living', 'bedrooms','bathrooms','lat','long','bed_bath_rooms']
features_of_model_3=features_of_model_2 + ['bedroom_squared','log_sqft_living','lat_plus_long']


#6
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
model_1=reg.fit(train_data[features_of_model_1], train_data['price'])
predicted_output_1=reg.predict(train_data[features_of_model_1])
p1=reg.predict(test_data[features_of_model_1])

#7
reg=LinearRegression()
model_2=reg.fit(train_data[features_of_model_2], train_data['price'])
predicted_output_2=reg.predict(train_data[features_of_model_2])
p2=reg.predict(test_data[features_of_model_2])

#8
reg=LinearRegression()
model_3=reg.fit(train_data[features_of_model_3], train_data['price'])
predicted_output_3=reg.predict(train_data[features_of_model_3])
p3=reg.predict(test_data[features_of_model_3])

#9
print(model_1.intercept_,'',model_1.coef_)
print(model_2.intercept_,'',model_2.coef_)
print(model_3.intercept_,'',model_3.coef_)

def get_resedual_sum_of_squares(y,y_hat):
    r=y-y_hat
    rs=r*r
    rss=rs.sum()
    return rss
rss_1=get_resedual_sum_of_squares(test_data['price'],p1)
print('rss for model1:', rss_1)

rss_2=get_resedual_sum_of_squares(test_data['price'],p2)
print('rss for model1:', rss_2)

rss_3=get_resedual_sum_of_squares(test_data['price'],p3)
print('rss for model1:', rss_3)