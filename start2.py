from Run import Network
import numpy as np
import pandas as pd



test_data = pd.read_csv(r'C:\Users\Mehdi\Desktop\MEHDI\Projects\Dl\regresion\house-prices-advanced-regression-techniques\test.csv')
train_data  = pd.read_csv(r'C:\Users\Mehdi\Desktop\MEHDI\Projects\Dl\regresion\house-prices-advanced-regression-techniques\train.csv')

features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]

X_train       = train_data[features]
y_train       = train_data["SalePrice"]
final_X_test  = test_data[features]


X_train      = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())


input_dim        = X_train.shape[1] # number of neurons in the input layer
n_neurons        =  5       # number of neurons in the first hidden layer
epochs           = 150       # number of training cycles

a=Network(input_size=input_dim,output_size=1,output_activision='tanh')
a.hiddenDenseLayers(hidden_layers=[(n_neurons,'leakyrelu')])
result=a.fit(x=X_train,target=y_train,epochs=1500,learning_rate=0.1
      ,batch_size=64,lossfunction='mean_squared_error',optimization='gradient_descent_with_momentum',iter=1,stochastic=True)
a.errorToepoch()
