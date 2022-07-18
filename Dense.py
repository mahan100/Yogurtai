from Layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weight =np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
        
    def forward(self,input):
        self.input=np.vstack(input)
        # print(input)
        # print(self.weight)
        return np.dot(self.weight,self.input)+self.bias   
    
    def backward(self,learning_rate,grad_err_output,optimization):
        grad_err_weight=np.dot(grad_err_output,self.input.T)
        self.updateParams(learning_rate,grad_err_weight,grad_err_output,optimization)
        grad_err_input=np.dot(self.weight.T,grad_err_output)
        return grad_err_input
  
    def updateParams(self,learning_rate,grad_err_weight,grad_err_output,optimization):
        self.weight-=optimization.learningWeightsFunction(learning_rate,grad_err_weight)
        self.bias-=optimization.learningBiasFunction(learning_rate,grad_err_output)
        