class Optimization():
    def __init__(self):  
        self.learning_weights_func = None
        self.learning_biases_func = None
    
    
class GradientDescent(Optimization):
    def __init__(self):  
        self.learning_weights_func = None
        self.learning_biases_func = None
        
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func
    
    
class Adam(Optimization):
    def __init__(self):  
        self.learning_weights_func = None
        self.learning_biases_func = None
        
    def learningWeightsFunction(self,learning_rate,grad_err_weight):
        self.learning_weights_func = learning_rate*grad_err_weight
        return self.learning_weights_func
    
    def learningBiasFunction(self,learning_rate,grad_err_output):
        self.learning_biases_func =learning_rate*grad_err_output
        return self.learning_biases_func
        