#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
pyoff.init_notebook_mode(connected=True)


# In[2]:


def prepare_data(timeseries,features):
    X,y=[],[]
    for i in range(len(timeseries)):
        endix=i+features
        if endix> len(timeseries)-1:
            break
        
        
        seq_x,seq_y=timeseries[i:endix],timeseries[endix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X),np.array(y)
               


# In[3]:


df=pd.read_csv('C:/Users/Administrator/Desktop/areebfolder/project/og/ForexMarket.csv')
df=df['price']
#type(df)


# In[4]:


df=df.values.tolist()


# In[ ]:


#l=46


# In[ ]:


#df=df[:-l]
#df


# In[ ]:


len(df)


# In[5]:


timeseries=df    #[2.4,4.8,6.4,8.8,10.4,12.8,14.4,16.8,18.4]
n_steps=50
X,y=prepare_data(timeseries,n_steps)
timeseries


# In[54]:


print(X,y)


# In[8]:


X.shape


# In[9]:


features=1
X=X.reshape((X.shape[0],X.shape[1],features))
X


# In[10]:


model=Sequential()
model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(n_steps,features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(X,y,epochs=100,verbose=1)


# In[11]:


#from array import array 
n=50
test=df[-n:]


# In[12]:




x_input=np.array(test)   
temp_input=list(x_input)
lst_output=[]
i=0
while(i<48):
    
    if(len(temp_input)>50):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape((1,n_steps,features))
        ymod=model.predict(x_input,verbose=0)
        temp_input.append(ymod[0][0])
        temp_input=temp_input[1:]
        lst_output.append(ymod[0][0])
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,features))
        ymod=model.predict(x_input,verbose=0)
        temp_input.append(ymod[0][0])
        lst_output.append(ymod[0][0])
        i=i+1

print(lst_output)

        


# In[13]:


df_predict = pd.DataFrame((lst_output),columns=['Prediction'])
df_predict


# In[14]:


days=pd.read_csv('C:/Users/Administrator/Desktop/areebfolder/project/og/Timezone.csv')
predicted_data = pd.concat([days,df_predict], axis=1)
predicted1=pd.DataFrame(predicted_data)
predicted1


# In[51]:


actual_data=pd.read_csv('C:/Users/Administrator/Desktop/areebfolder/project/og/today.csv')
actual_data=actual_data['price']
k=actual_data
actual_dt=pd.DataFrame(k)
actual_dt

new=pd.concat([predicted1,actual_dt],axis=1)
new


# In[50]:



plot_data = [
    go.Scatter(
        x=predicted1.Time,
        y=new.price,
        name='actual'
    ),
    #go.Scatter(
        #x=predicted1.Time,
        #y=new.Prediction,
        #name='prediction'
    #)
]

plot_layout = go.Layout(
        title='Actual Forex market'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[49]:


plot_data = [
    go.Scatter(
        x=predicted1.Time,
        y=new.Prediction,
        name='prediction'
    ),
]
plot_layout = go.Layout(
        title='Predicted Forex market'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:




