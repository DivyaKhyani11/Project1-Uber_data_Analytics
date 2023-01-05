#!/usr/bin/env python
# coding: utf-8

# #  Data Preparation for Analysis
# 
# 
# 

# In[1]:


##importing all the necessary packages 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


## read data 
uber_15=pd.read_csv('D:\data analytics projects/uber-raw-data-janjune-15.csv')


# In[3]:


##view of the data
uber_15.head(2)


# In[4]:


## getting dimensions of data
uber_15.shape


# In[5]:


## getting count of total duplicated observations in your data
uber_15.duplicated().sum()


# In[6]:


## deleting all the duplicated observations
uber_15.drop_duplicates(inplace=True)


# In[7]:


uber_15.shape


# # 1. which month has maximum uber pickups in NY city ?

# In[8]:


### data-type of your features
uber_15.dtypes


# we can see that "Pickup_date" is a object data type,
# Therefore, we have to convert this datatype into date-time becuase at the end we have to extract Derived attributes.
# 
# -- For this we require pandas to_datetime to convert object data type to datetime dtype.

# In[9]:


uber_15['Pickup_date']=pd.to_datetime(uber_15['Pickup_date'], format ='%Y-%m-%d %H:%M:%S' )


# In[10]:


uber_15['Pickup_date'].dtype


# In[11]:


## extracting month from 'Pickup_date'..
uber_15['month']=uber_15['Pickup_date'].dt.month


# In[12]:


uber_15['month'].value_counts().plot(kind='pie')


# # 2.  Calculating total trips for each month & each weekday

# In[13]:


# extracting dervied features (weekday ,day ,hour ,month ,minute) from 'Pickup_date'

uber_15['weekday']=uber_15['Pickup_date'].dt.day_name()
uber_15['day']=uber_15['Pickup_date'].dt.day
uber_15['hour']=uber_15['Pickup_date'].dt.hour
uber_15['month']=uber_15['Pickup_date'].dt.month
uber_15['minute']=uber_15['Pickup_date'].dt.minute


# In[14]:


#separte columns created for month, weekday, hour, and minute 
uber_15.head(2)


# In[15]:


#grouping month and weekday
temp=uber_15.groupby(['month','weekday'],as_index=False).size()
temp.head()


# In[16]:


temp.tail()


# In[17]:


temp['month'].unique()


# In[18]:


# storing as key value pair
dict_month={1:'Jan', 2:'Feb', 3:'March', 4:'april', 5:'May', 6:'June'}


# In[19]:


temp['month']=temp['month'].map(dict_month)


# In[20]:


temp['month']


# In[21]:


type(uber_15.groupby(['month','weekday']).size())


# In[22]:


temp


# In[23]:


## create grouped bar chart ..

plt.figure(figsize=(14,10)) #customizing the size of barplot
sb.barplot(x='month',y='size',hue='weekday',data=temp)


# # 3. Analysing hourly demand of Uber in NY CITY

# In[24]:


uber_15.groupby(['weekday','hour']).count()


# In[25]:


summary=uber_15.groupby(['weekday','hour'],as_index=False).size()


# In[26]:


summary


# In[27]:


## pointplot between 'hour' & 'size' for all the weekdays.

plt.figure(figsize=(14,10))
sb.pointplot(x='hour',y='size',hue='weekday',data=summary)


# '''
# It's interesting to see that Saturday and Sunday exhibit similar demand throughout the late night/morning/afternoon, 
# but it exhibits opposite trends during the evening. In the evening, Saturday pickups continue to increase throughout the evening,
# but Sunday pickups takes a downward turn after evening..
# 
# We can see that there the weekdays that has the most demand during the late evening is Friday and Saturday, 
# which is expected, but it can also be observed that Thursday nights also exhibit similar trends as Friday and Saturday nights.
# 
# It seems like New Yorkers are starting their 'weekends' on Thursday nights :)
# '''
# 

# # 4. Analysing most active Uber- Base number

# In[28]:


#read another data set as we need active user deatils
uber_foil=pd.read_csv(r'D:\data analytics projects/Uber-Jan-Feb-FOIL.csv')


# In[29]:


#to look at data in csv file 
uber_foil.head()


# In[30]:


#installing chart studio
get_ipython().system('pip install chart_studio')


# In[31]:


#install plotly
get_ipython().system('pip install plotly')


# In[32]:


### establishing the entire set-up of Plotly.
## graph_objs and express are sub modules that contain couple of visualization tools!
## download_plotlyjs ,plot ,iplot ,init_notebook_mode will help in showcasing the visuals
## for visualization in i python notebook, the variable connected is set as true for init_notebook_mode.

import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs ,plot ,iplot ,init_notebook_mode
init_notebook_mode(connected=True)


# In[33]:


#box plot - only 5 point summary, no distribution
px.box(x='dispatching_base_number',y='active_vehicles' ,data_frame=uber_foil)


# In[34]:


#voilin plot- 5 point summary + distribution
px.violin(x='dispatching_base_number',y='active_vehicles' ,data_frame=uber_foil)


# # Collecting entire data & Making it ready for the Data Analysis.

# In[35]:


import os


# In[36]:


files=os.listdir(r'D:\data analytics projects\Uber_datasets')


# In[37]:


files


# In[38]:


#to get the data in april to sept month
my_files=os.listdir(r'D:\data analytics projects\Uber_datasets')[-7:]


# In[39]:


my_files


# In[40]:


path=r'D:\data analytics projects\Uber_datasets'

#blank dataframe
final=pd.DataFrame()

for file in my_files:
    current_df=pd.read_csv(path+'/'+file,encoding='utf-8')
    final=pd.concat([current_df,final])


# In[41]:


final.shape


# In[42]:


final.head(2)


# In[43]:


### first lets find total observations where we have duplicate values.
final.duplicated().sum()


# In[44]:


## drop duplicate values ..
### By default, it removes duplicate rows based on all columns.
### To remove duplicates on specific column(s), use subset parameter of 'drop_duplicates()'

### by-default , keep='first which says it will keep first occurence of duplicates...'

final.drop_duplicates(inplace=True)


# In[45]:


#lets check the shape of the data now
final.shape


# ### The dataset contains information about the Datetime, Latitude, Longitude and Base of each uber ride 
# Date/Time : The date and time of the Uber pickup
# 
# Lat : The latitude of the Uber pickup
# 
# Lon : The longitude of the Uber pickup
# 
# Base : The TLC base company code affiliated with the Uber pickup
# 
#     The Base codes are for the following Uber bases:
#     B02512 : Unter
#     B02598 : Hinter
#     B02617 : Weiter
#     B02682 : Schmecken
#     B02764 : Danach-NY
#     ->> The globe is split into an imaginary 360 sections from both top to bottom (north to south) and 180 sections from side to side (west to east). The sections running from top to bottom on a globe are called longitude, and the sections running from side to side on a globe are called latitude.
#     ->> Latitude is the measurement of distance north or south of the Equator.
#     ->> Every location on earth has a global address. Because the address is in numbers, people can communicate about location no matter what language they might speak. A global address is given as two numbers called coordinates. The two numbers are a location's latitude number and its longitude number ("Lat/Long").

# # 5. Analysis to find the locations of New York City we are getting rush 
### i.e; whereever we have more data-points or more density, it means more rush!
# In[46]:


rush_uber=final.groupby(['Lat','Lon'],as_index=False).size()


# In[47]:


rush_uber


# In[48]:


#installing folium package for map functions 
get_ipython().system('pip install folium')


# In[49]:


import folium


# In[50]:


#to look at the world map
basemap=folium.Map()


# In[51]:


from folium.plugins import HeatMap


# In[52]:


HeatMap(rush_uber).add_to(basemap)


# In[53]:


basemap


# We can see a number of hot spots here. Midtown Manhattan is clearly a huge bright spot.
# & these are made from Midtown to Lower Manhattan.
# Followed by Upper Manhattan and the Heights of Brooklyn.

# # 6. Perform pairwise analysis to find rush in hour and weekday

# In[54]:


final.head(2)


# In[55]:


### converting 'Date/Time' feature into date-time..

final['Date/Time']=pd.to_datetime(final['Date/Time'],format ='%m/%d/%Y %H:%M:%S')


# In[56]:


### extracting 'weekday' & 'hour' from 'Date/Time' feature..

final['weekday']=final['Date/Time'].dt.day
final['hour']=final['Date/Time'].dt.hour


# In[57]:


final.head(3)


# In[58]:


#unstack is used as the data type is series 
pivot=final.groupby(['weekday','hour']).size().unstack()


# In[59]:


type(final.groupby(['weekday','hour']).size())


# In[60]:


### pivot table determines the Rows*columns & has value in each cell !
pivot


# In[61]:


### styling dataframe
#the higher number is highlighted for us to differentiate
pivot.style.background_gradient()


# # Automating my analysis

# In[62]:


## creating a user-defined function..

def generate_pivot_table(df,col1,col2):
    pivot=df.groupby([col1,col2]).size().unstack()
    return pivot.style.background_gradient()


# In[63]:


final.columns


# In[64]:


generate_pivot_table(final,'weekday','hour')


# In[ ]:




