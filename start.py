from Run import Network
import numpy as np
import pandas as pd
 
# arr=np.random.randn(100,3)
# target=np.random.randn(100,1)

arr=np.array([[1,0,1],[0,1,1],[0,0,0],[1,1,0]])
df=pd.DataFrame(arr,columns=['x1','x2','target'])
# print(df.iloc[:,:-1],df.iloc[:,:-1])
# a=df.iloc[:,:-1]
# b=df.iloc[:,-1]
# ss=pd.concat([a,b],axis=1)
# # print(ss)
# for index,row in ss.iterrows():
#       l=ss.iloc[index,:]
#       f=l.to_numpy()
#       print(f)
#       print('*')
      
# print(np.random.randint(4,size=(3,2)))

a=Network(input_size=2,output_size=1,output_activision='tanh')
a.hiddenDenseLayers(hidden_layers=[(6,'tanh')])
a.fit(x=df.iloc[:,:-1],target=df.iloc[:,-1],epochs=1000,learning_rate=0.1
      ,batch_size=1,lossfunction='mean_squared_error',optimization='gradient_descent',iter=1,stochastic=True)
a.errorToepoch()