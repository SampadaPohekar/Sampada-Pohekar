#!/usr/bin/env python
# coding: utf-8

# Sampada Pohekar

# Task 1: Stock Market Prediction And Forecasting Using Stacked LSTM

# Dataset: https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv
# 

# In[13]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
dataset_link = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"


# In[14]:


df=pd.read_csv(dataset_link)
df


# In[15]:


df.head()


# In[16]:


df.describe()


# In[17]:


df.shape


# In[18]:


df.info()


# In[19]:


df.head()


# In[20]:


df.isnull().sum()


# In[21]:


new_df=pd.read_csv(dataset_link,parse_dates=["Date"])
df.head()


# # Sorting Data

# In[22]:


new_df['Date'] = pd.to_datetime(new_df['Date'],errors='coerce')
print(type(new_df.Date[0]))


# In[23]:


new_df.sort_values(by=['Date'],inplace=True,ascending=True)
new_df.Date.head()


# # Data Visualization

# In[24]:


fig,ax=plt.subplots()
ax.scatter(new_df.Date,new_df.Close)


# In[25]:


new_df.reset_index(inplace=True)
new_df


# # Univariate Analysis of Closing Price

# In[26]:


close_df=new_df['Close']
close_df


# In[27]:


close_df.size


# In[28]:


close_df.shape


# In[29]:


close_df.describe()


# # Min Max Scaler

# In[30]:


scaler=MinMaxScaler(feature_range=(0,1))
close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))
close_df


# # Train and Test Split

# In[31]:


training_size=int(len(close_df)*0.7)
test_size=len(close_df)-training_size
train_data,test_data=close_df[0:training_size,:],close_df[training_size:len(close_df),:1]


# In[32]:


train_data.shape,close_df.shape


# In[33]:


test_data.shape


# In[34]:


type(test_data)


# # Data Preprocessing

# In[35]:


import numpy as np
def create_dataset(dataset,time_step=1):
    dataX=[]
    dataY=[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX),np.array(dataY)
time_step = 100
#reshaping into tuples
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# In[36]:


x_test, y_test = create_dataset(test_data, time_step)


# In[37]:


y_train.shape


# In[38]:


x_train.shape


# In[39]:


print(x_test.shape),print(y_test.shape)


# In[40]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# # Creating the stacked LSTM Model

# In[3]:


pip install tensorflow


# In[41]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))#input layer with 50 neurons
model.add(LSTM(50,return_sequences=True))#hidden layers with 50 neurons
model.add(LSTM(50))
model.add(Dense(1))#output layer
model.compile(loss='mean_squared_error',optimizer='adam')


# In[42]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics='acc')


# In[43]:


model.summary()


# In[46]:


model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs =100, batch_size = 64, verbose = 1);


# Evaluation of constructed LSTM model

# In[48]:


#ploting loss of our trained model
loss=model.history.history['loss']
plt.plot(loss)
plt.xlabel("No.of iteration...")
plt.ylabel("loss value...")
plt.title("variation of loss value with No. of iteration",color="orange",fontsize=16,fontweight="bold");


# The above graph shows that loss has been decreased significantly with the increase in iteration,thus model is well trained

# Evaluation of our constructed model on train and test data

# In[50]:


train_predict1=model.predict(x_train)


# In[52]:


test_predict1=model.predict(x_test)


# In[53]:


#Transformback to original form
train_predict1=scaler.inverse_transform(train_predict1)
test_predict1=scaler.inverse_transform(test_predict1)


# In[54]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict1))


# In[56]:


math.sqrt(mean_squared_error(y_test,test_predict1))


# In[57]:


close_df


# In[58]:


train_predict1.shape


# # Plotting

# In[59]:


# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(close_df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict1)+look_back, :] = train_predict1

# shift test predictions for plotting
testPredictPlot = np.empty_like(close_df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict1)+(look_back*2)+1:len(df)-1, :] = test_predict1


# In[60]:


# plot baseline and predictions
plt.figure(figsize=(10,5))
plt.plot(scaler.inverse_transform(close_df))
plt.title("variation of actual dataset",color="orange",fontsize=16,fontweight="bold")
plt.xlabel("data index number")
plt.ylabel("closing price")


# In[62]:


#plot of predictions on closing price made by our model on training dataset vs actual closing price
plt.plot(scaler.inverse_transform(close_df))
plt.plot(trainPredictPlot)
plt.title("variation of predicted trained dataset(orange) and actual dataset(lightblue)",color="orange",fontsize=16,fontweight="bold")
plt.xlabel("data index number")
plt.ylabel("closing price")


# In[63]:


#plot of prediction on closing price made by our model on training dataset vs actual closing price
#vs prediction on closing price made by our model on test dataset
plt.plot(scaler.inverse_transform(close_df))
plt.plot(trainPredictPlot)
plt.title("variation of predicted tested dataset(green) and actual dataset(lightblue)",color="orange",fontsize=16,fontweight="bold")
plt.xlabel("data index number")
plt.ylabel("closing price")
plt.plot(testPredictPlot)
plt.show()


# since our model trained with first 100 data inputs and start predicting from 101 onwards
# so orange starts from 101 dataindex no. , similar logic for test data
# 
# 
# light blue=-actual closing price
# 
# 
# orange=prediction on closing price made by our model on training dataset
# 
# 
# green=prediction on closing price made by model on test dataset
