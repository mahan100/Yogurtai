from re import X
from pandas import array
from Layer import Layer
import numpy as np

class Activision(Layer):
    def __init__(self, activision, activision_prim):
        self.activision =activision 
        self.activision_prim = activision_prim
    def forward(self,input):
        self.input=input
        input=input.tolist()
        output=self.activision(input)
        return np.array(output)
        
    def backward(self,learning_rate,grad_err_output,optimization):
        input=self.input
        input=input.tolist()
        activision_output=self.activision_prim(input[0])
        grad_err_input=grad_err_output*activision_output
        return np.array(grad_err_input)
    
class Tanh(Activision):
    def __init__(self):
        function=lambda x:[np.tanh(i) for i in x]
        function_prim=lambda x:[1-np.tanh(i)**2 for i in x]
        super().__init__(function,function_prim)

class Relu(Activision):
    def __init__(self):
        function=lambda x:[np.max(i,0) for i in x]
        function_prim=lambda x:[1 if i>=0 else 0 for i in x]
        super().__init__(function,function_prim)   
class LeakyRelu(Activision):
    def __init__(self):
        def func(x):
            l=[]
            for i in range(len(x)):
                if i>=0:
                    l.append(i)
                else:
                    l.append((i*0.01))
            return l
        def func_prim(x):
            l=[]
            for i in range(len(x)):
                if i>=0:
                    l.append(1)
                else:
                    l.append(0.01)
            return l
        function,function_prim=func,func_prim
        
        super().__init__(function,function_prim)  
             
# class BinaryClassificaton(Activision):
#     def __init__(self):
#         function=lambda x:1 if x>0.5 else 0
#         function_prim=lambda x:0
#         super().__init__(function,function_prim)
              