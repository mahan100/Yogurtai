from Optimization import Optimization
from Layer import Layer
import numpy as np
class Dense(Optimization):
    
    def __init__(self,input_size,output_size):
        super().__init__()
        self.weight =np.random.uniform(0,input_size,size=(output_size,input_size))
        self.bias = np.random.uniform(0,0.0001,size=(output_size,1))
        self.output_size=output_size
        self.input_size=input_size

    def forward(self,input):
        self.input=input.reshape(self.input_size,1)
        output=np.dot(self.weight,self.input)+self.bias
        return output

    def backward(self,learning_rate,grad_err_output,optimization):
        grad_err_output=grad_err_output.reshape(self.output_size,1)
        self.input=self.input.reshape(1,self.input_size)
        grad_err_weight=np.dot(grad_err_output,self.input)
        self.optimizationFunction(optimization)
        self.weight,self.bias=self.opt_func(self,learning_rate,grad_err_weight,grad_err_output)
        grad_err_input=np.dot(self.weight.T,grad_err_output)
        return self.weight,self.bias,grad_err_input
        
    def optimizationFunction(self,opt):
       opt=opt.lower()
       if opt == 'gradient_descent':
           self.opt_func=super().gradientDescent  
       elif opt == 'gradient_descent_with_momentum':
           self.opt_func=super().gradientDescentWithMomentum
       elif opt == 'adagrad':
           self.opt_func=super().adagrad
       elif opt == 'rmsprop':
           self.opt_func=super().rmsProp
       elif opt == 'adadelta':
           self.opt_func=super().adaDelta
       elif opt == 'adam':
           self.opt_func=super().adam                  
       else:
           raise ValueError('Loss Function Did Not Define Correctly')    