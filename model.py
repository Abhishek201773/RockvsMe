
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df=pd.read_csv('sonar.csv',header=None)
df.head()
df.describe
df.isna().sum()
df[60].value_counts()
X=df.drop(columns=60,axis=1)
Y=df[60]
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=1,stratify=Y)
"""LOGISTIC **REGRESSION**
"""
model=LogisticRegression()
model.fit(X_train,Y_train)
pickle.dump(model, open('model.pkl', 'wb'))
X_train_pred=model.predict(X_train)
X_accuracy=accuracy_score(X_train_pred,Y_train)
print("accuracy Score   ",X_accuracy*100)
X_test=model.predict(X_test)
X_accuracy_test=accuracy_score(X_test,Y_test)
print("accuracy on test  ",X_accuracy_test)
input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')