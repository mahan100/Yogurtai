from pickletools import float8
import numpy as np
from pandas import array
from functools import wraps
class Optimization():
    def __init__(self):
        self.B=0
        self.V=0
        self.beta=0.9
        
    
    def updateParams(func):
        @wraps(func)
        def wraper(self,*args,**kwargs):
                weight_grad_term,bias_grad_term=func(*args,**kwargs)  
                self.weight -= np.multiply(args[1],weight_grad_term)
                self.bias -=np.multiply(args[1],bias_grad_term)
                return self.weight,self.bias
        return wraper      

    
    @updateParams
    def gradientDescent(self,*args,**kwargs):
            weight_grad_term,bias_grad_term=args[1],args[2]
            return weight_grad_term,bias_grad_term
    
    @updateParams
    def gradientDescentWithMomentum(self,*args,**kwargs):
            self.V=np.multiply(self.beta,self.V)+np.multiply((1-self.beta),args[1])  
            self.B=np.multiply(self.beta,self.B)+np.multiply((1-self.beta),args[2])
            weight_moving_average,bias_moving_average=self.V,self.B
            return np.array(weight_moving_average),np.array(bias_moving_average)

    def adagrad(self,learning_rate,grad_err_output):
       pass

    def rmsProp(self,learning_rate,grad_err_output):
       pass 

    def adaDelta(self,learning_rate,grad_err_output):
      pass       

    def adam(self,learning_rate,grad_err_output):
         pass

                