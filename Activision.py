from Layer import Layer
import numpy as np

class Activision(Layer):
    def __init__(self, activision, activision_prim):
        self.activision =activision 
        self.activision_prim = activision_prim
    def forward(self,input):
        self.input=input
        return self.activision(self.input)
    def backward(self,learning_rate,grad_err_output,optimization):
        grad_err_input=np.multiply(grad_err_output,self.activision_prim(self.input))       
        return grad_err_input
    
class Tanh(Activision):
    def __init__(self):
        function=lambda x:np.tanh(x)
        function_prim=lambda x:1-np.tanh(x)**2
        super().__init__(function,function_prim)

class Relu(Activision):
    def __init__(self):
        function=lambda x:np.max(x,0)
        function_prim=lambda x:1 if x>0 else 0
        super().__init__(function,function_prim)        

              