from tkinter import END
import numpy as np

class LossFunction():
    def __init__(self):
        self.function = None
        self.function_prim = None
        
    def lossFunction(self):
        pass         
    def lossFunction_prim(self):
        pass
    
class MeanScoreError(LossFunction):
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.power((y_true-y_pred),2)
        self.function=np.float(np.sum(self.function)/self.y_size)
        return self.function         
    def lossFunction_prim(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function_prim=2*((y_pred-y_true))
        self.function_prim=np.float(np.sum(self.function_prim)/self.y_size)
        return self.function_prim       
    
class RootMeanScoreError(LossFunction):
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.power((y_true-y_pred),2)
        self.function=np.sum(self.function)/self.y_size
        self.function=np.float(np.sqrt(self.function))
        return self.function         
    def lossFunction_prim(self,y_true,y_pred):
        self.function_prim=1/(2*MeanScoreError().lossFunction(y_true,y_pred))
        self.function_prim=self.function*MeanScoreError().lossFunction_prim(y_true,y_pred)
        return self.function_prim    

class MeanAbsoluteError(LossFunction):
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.abs((y_true-y_pred,2))
        self.function=np.float(np.sum(self.function)/self.y_size)
        return self.function         
    def lossFunction_prim(self,y_true,y_pred):
        self.function_prim=np.float(np.sum((y_pred-y_true)))
        if self.function >0:self.function=-1 
        else:1
        return self.function_prim             
    
class LogLoss(LossFunction):
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.int(np.power(y_true-y_pred,2))
        return self.function         
    def lossFunction_prim(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function_prim=2*(y_pred-y_true)/np.size(self.y_size)
        return self.function_prim         
    
class SoftMax(LossFunction):
    def lossFunction(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function=np.int(np.power(y_true-y_pred,2))
        return self.function         
    def lossFunction_prim(self,y_true,y_pred):
        self.y_size=len(y_true)
        self.function_prim=2*(y_pred-y_true)/np.size(self.y_size)
        return self.function_prim    