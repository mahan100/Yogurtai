from Layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weight =np.random.uniform(1.0,input_size,size=(output_size,input_size))
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
        grad_err_input=np.dot(self.weight.T,grad_err_output)
        self.updateParams(learning_rate,grad_err_weight,grad_err_output,optimization)
        return self.weight,self.bias,grad_err_input
  
    def updateParams(self,learning_rate,grad_err_weight,grad_err_output,optimization):
        self.weight-=optimization.learningWeightsFunction(learning_rate,grad_err_weight)
        self.bias-=optimization.learningBiasFunction(learning_rate,grad_err_output)
        