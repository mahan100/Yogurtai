from distutils.log import error
from lib2to3 import refactor
from sklearn import datasets
from sklearn.utils import shuffle
from Dense import Dense
from Activision import Tanh,Relu
from LossFunction import MeanScoreError,RootMeanScoreError,MeanAbsoluteError,SoftMax
from Optimization import GradientDescent,Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Network():
    def __init__(self,input_size,output_size,activision):
        self.input_size=input_size
        self.output_size=output_size
        self.activision=activision
        self.net=[]
        self.loss_obj=None
        self.opt_obj=None
        self.result_epoch_mean=[]


    def hiddenDenseLayers(self,hidden_layers):
        input=self.input_size
        for i in hidden_layers:
            self.net.append(Dense(input,i[0]))
            input=i[0]
            self.activisionFunction(i[1])
        self.net.append(Dense(input,self.output_size))
        self.activisionFunction(self.activision)
          
    def activisionFunction(self,i):
        activision_name=i.lower()
        if activision_name == 'relu':
            self.net.append(Relu())
        elif activision_name =='tanh':
            self.net.append(Tanh())
        else:
            raise ValueError('Activision Function Did Not Define Correctly')
        
    def lossFunction(self,loss):
        loss=loss.lower()
        if loss == 'mean_score_error':
            self.loss_obj=MeanScoreError()
        elif loss == 'root_mean_score_error':
            self.loss_obj=RootMeanScoreError()
        elif loss == 'mean_absolute_error':
            self.loss_obj=MeanAbsoluteError()
        else:
            raise ValueError('Loss Function Did Not Define Correctly')

    def optimizationFunction(self,opt):
        opt=opt.lower()
        if opt == 'gradient_descent':
            self.opt_obj=GradientDescent()
        elif opt == 'adam':
            self.opt_obj=Adam()
        else:
            raise ValueError('Loss Function Did Not Define Correctly')
    
    def fit(self,x,target,epochs,learning_rate,batch_size,lossfunction,optimization):
        self.X=x
        self.target=target
        self.EPOCHS=epochs
        self.LEARNING_RATE=learning_rate
        self.BATCH_SIZE=batch_size
        self.lossfunction=lossfunction
        self.optimization=optimization
        
        self.lossFunction(self.lossfunction)
        self.optimizationFunction(self.optimization)
                
        for epoch in range(self.EPOCHS):
            result_list=[]
            error=0
            error_prim=0
            iteration=self.X.shape[0]/self.BATCH_SIZE
            iteration=np.floor(iteration)
            iteration=int(iteration)    
            
            for iter in range(iteration):
                start=self.BATCH_SIZE*iter
                end=(iter+1)*self.BATCH_SIZE
                output_iter_list=[]

                
                for x,y in zip(self.X[start:end],self.target[start:end]):
                    output=x
                    for layer in self.net:
                        output=layer.forward(output)
                    output_iter_list.append(output)
                    
                output_iter_array=np.array(output_iter_list)
                error=self.loss_obj.lossFunction(self.target[start:end],output_iter_array)
                error_prim =self.loss_obj.lossFunction_prim(self.target[start:end],output_iter_array)
                             
                grad_err_input=error_prim
                for back_layer in reversed(self.net):
                    grad_err_input=back_layer.backward(self.LEARNING_RATE,grad_err_input,self.opt_obj)   
                                     
                result={'epoch':epoch,'iter':iter,'error':error}
                result_list.append(result)
            
            result_df=pd.DataFrame(result_list,columns=['epoch','iter','error'])  
            result_mean=result_df.loc[result_df['epoch']==epoch]
            self.result_epoch_mean.append(result_mean['error'].mean())
            
            print(f'***********epoch={epoch}      error={self.result_epoch_mean[epoch]}***************')        
    
    def errorToepoch(self):    
        plt.plot(range(self.EPOCHS),self.result_epoch_mean)
        plt.show()
        
        
                   
                