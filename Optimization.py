from pickletools import float8
import numpy as np
class Optimization():
    def __init__(self):  
        pass
class GradientDescent(Optimization):
           
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        learning_weights_func = np.multiply(learning_rate,grad_err_weight)
        return learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        learning_biases_func =learning_rate*grad_err_output
        return learning_biases_func
    
class StochasticGradientDescentWithMomentum(Optimization):
           
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        learning_weights_func = np.multiply(learning_rate,grad_err_weight)
        return learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        learning_biases_func =learning_rate*grad_err_output
        return learning_biases_func
    
class Adagrad(Optimization):
            
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func
    
class RmsProp(Optimization):
            
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func    
    
class AdaDelta(Optimization):
            
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func            
    
class Adam(Optimization):
            
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func
        