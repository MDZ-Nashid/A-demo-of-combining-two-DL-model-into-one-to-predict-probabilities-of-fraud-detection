#SOM

#importing libraries
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
ms = MiniSom(x=10, y=10, input_len=15,learning_rate=0.5,sigma =1)
ms.random_weights_init(X)
ms.train_random(data=X, num_iteration=100)

from pylab import pcolor, bone, colorbar, plot, show
bone()
pcolor(ms.distance_map().T)
markers = ['o','s']
colors = ['r','g']
colorbar()
for i,x in enumerate(X):
    w = ms.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mapping = ms.win_map(X)
frauds = np.concatenate( (mapping[(6,6)], mapping[(1,1)]),axis = 0)
frauds = sc.inverse_transform(frauds)


#ann

customers = dataset.iloc[:,1:].values
is_fraud = np.zeros(len(customers))

for i in range(len(customers)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1
        

import tensorflow as tf 
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 2,kernel_initializer ='uniform', activation = 'relu', input_dim = 15))


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

ann.fit(customers,is_fraud, batch_size = 1, epochs = 2)

#predicting probabilites of frauds


y_pred=ann.predict(customers)


y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)

y_pred = y_pred[y_pred[:,1].argsort()]
