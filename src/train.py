from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

import pandas as pd
from config import TARGET

data = pd.read_csv('../Data/admission_data.csv')
X= data.drop([TARGET], axis=1)
y= data[TARGET] 

X_train,X_test ,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

rmse = mse(y_test, y_pred, squared=False)

print(rmse)