from tkinter import END
import numpy as np
from pandas import array

class LossFunction():
    def __init__(self):
        self.function = None
        self.function_prim = None
        
    def lossFunction(self):
        pass         
    def lossFunction_prim(self):
        pass
    
class MeanSquaredError(LossFunction):
    
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.power((y_true-y_pred),2)
        self.function=np.float(np.sum(self.function)/self.y_size)
        return self.function         
    
    def lossFunction_prim(self,y_true,y_pred,X):
        self.y_size=len(y_true)
        self.function_prim=-2*((y_true-y_pred))
        self.function_prim=np.float(np.sum(self.function_prim)/self.y_size)
        return np.array(self.function_prim)
    
class RootMeanSquaredError(LossFunction):
    
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.power((y_true-y_pred),2)
        self.function=np.sum(self.function)/self.y_size
        self.function=np.float(np.sqrt(self.function))
        return self.function         
    
    def lossFunction_prim(self,y_true,y_pred,X):
        self.function_prim=1/(2*MeanSquaredError().lossFunction(y_true,y_pred))
        self.function_prim=self.function*MeanSquaredError().lossFunction_prim(y_true,y_pred,X)
        return self.function_prim    

class MeanAbsoluteError(LossFunction):
    
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.abs((y_true-y_pred,2))
        self.function=np.float(np.sum(self.function)/self.y_size)
        return self.function         
    
    def lossFunction_prim(self,y_true,y_pred,X):
            
        self.function_prim=np.float(np.sum((y_pred-y_true)))
        if self.function > 0 : self.function=-1 
        else:1
        return self.function_prim             
    
class LogLoss(LossFunction):
    
    def lossFunction(self,y_true,y_pred):
        probability=np.exp(y_pred)/(1+np.exp(y_pred))
        if probability==np.array([0.]):
            probability=np.array([0.001])
        elif probability>=np.array([1.]):
            probability=np.array([0.999])
        elif probability==np.array([-0.]):
            probability=np.array([-0.001])
        elif probability<=np.array([-1.]):
            probability=np.array([-0.999]) 

        logloss_first_term=(1-y_true)*np.log(1-probability)
        logloss_second_term=y_true*np.log(probability)
        logloss=-(logloss_first_term+logloss_second_term)
        self.binary_classification_result=1 if logloss>0.5 else 0
        return logloss
    
    def lossFunction_prim(self,y_true,y_pred,X):
        probability=np.exp(y_pred)/(1+np.exp(y_pred))
        if probability==np.array([0.]):
            probability=np.array([0.001])
        elif probability>=np.array([1.]):
            probability=np.array([0.999])
        elif probability==np.array([-0.]):
            probability=np.array([-0.001])
        elif probability<=np.array([-1.]):
            probability=np.array([-0.999]) 
        logloss_prim_first_term=-np.log(1-probability)
        logloss_prim_second_term=-np.log(probability)
        logloss_prim=logloss_prim_first_term+logloss_prim_second_term
        return logloss_prim.reshape((1,1))
                
class SoftMax(LossFunction):
    def lossFunction(self,y_true,y_pred):
        pass       
    def lossFunction_prim(self,y_true,y_pred):
        pass