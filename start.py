import Run as n
from Run import Network
import numpy as np
 
x=np.random.randn(100,3)
target=np.random.randn(100,1)

a=Network(input_size=3,output_size=1,activision='tanh')
a.hiddenDenseLayers(hidden_layers=[(3,'tanh')])
a.fit(x=x,target=target,epochs=100,learning_rate=0.1
      ,batch_size=1,lossfunction='root_mean_score_error',optimization='gradient_descent')
a.errorToepoch()