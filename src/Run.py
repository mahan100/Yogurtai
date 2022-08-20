from Stats import Stats
from random import random
from sklearn import datasets
from sklearn.utils import shuffle
from Dense import Dense
from Activision import Tanh,Relu,LeakyRelu
from LossFunction import LogLoss,MeanSquaredError,RootMeanSquaredError,MeanAbsoluteError,SoftMax
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Network():
    def __init__(self,input_size,output_size,output_activision):
        self.input_size=input_size
        self.output_size=output_size
        self.activision=output_activision
        self.net=[]
        self.obj_loss=None
        self.obj_opt=None
        self.result_epoch_mean=[]

    #making hidden layers
    def hiddenDenseLayers(self,hidden_layers):
        input=self.input_size
        for i in hidden_layers:
            self.net.append(Dense(input,i[0]))
            input=i[0]
            self.obj_act(i[1])
        self.net.append(Dense(input,self.output_size))
        self.obj_act(self.activision)

    #selecting and making activision object
    def obj_act(self,i):
        activision_name=i.lower()
        if activision_name == 'relu':
            self.net.append(Relu())
        elif activision_name =='tanh':
            self.net.append(Tanh())
        elif activision_name =='leakyrelu':
            self.net.append(LeakyRelu())
        else:
            raise ValueError('Activision Function Did Not Define Correctly')
    
    #select and make lossfunction object  
    def lossFunction(self,loss):
        loss=loss.lower()
        if self.output_size==1:
            if loss == 'mean_squared_error':
                self.obj_loss=MeanSquaredError()
            elif loss == 'root_mean_squared_error':
                self.obj_loss=RootMeanSquaredError()
            elif loss == 'mean_absolute_error':
                self.obj_loss=MeanAbsoluteError()
            elif loss == 'logloss':
                self.obj_loss=LogLoss()
            else:
                raise ValueError('Loss Function Or Output Size Did Not Define Correctly')    
        else:        
            if loss == 'softmax':
                self.obj_loss=SoftMax()        
            else:
                raise ValueError('Loss Function Or Output Size Did Not Define Correctly')
    
    #determine iteration base on batci_size
    def iterBatch(self):
        self.iteration=self.X.shape[0]/self.BATCH_SIZE
        self.iteration=np.floor(self.iteration)
        self.iteration=int(self.iteration)
    #train model
    def fit(self,x:pd.DataFrame,target:pd.DataFrame,epochs=1000,
            learning_rate=0.1,batch_size=32,iter=1,
            lossfunction='mean_score_error',optimization='adam',stochastic=False):
        
        self.X=x
        self.target=target
        self.EPOCHS=epochs
        self.LEARNING_RATE=learning_rate
        self.BATCH_SIZE=batch_size
        self.lossfunction=lossfunction
        self.optimization=optimization
        self.iter=iter
        self.stochastic=stochastic
        self.result_list=[]
        
        self.lossFunction(self.lossfunction)
        self.df=pd.concat([self.X,self.target],axis=1)        

        if self.optimization.lower()=='gradient_descent':       
            self.BATCH_SIZE=1
        #epoch
        for epoch in range(self.EPOCHS):
            result_temp=[]
            error=0
            error_prim=0
            rand_num_list=[]
            if self.stochastic:
                self.iteration=self.iter
            else:
                self.iterBatch()
                
            for iter in range(self.iteration):
                output_iter_list=[]
                    
                #feed forward
                if self.stochastic == False:
                    start=self.BATCH_SIZE*iter
                    end=(iter+1)*self.BATCH_SIZE
                    df=self.df.iloc[start:end,:]
                    
                    for index,row in df.iterrows():
                        output=self.df.iloc[index,:-1]
                        output=output.to_numpy()
                        for layer in self.net:
                            output=layer.forward(output)
                        output_iter_list.append(output)
                else:
                    while True:   
                        count=0 
                        rand_num=np.random.randint(0,len(target))
                        if rand_num in rand_num_list:
                            count+=1
                            if count >1:
                                break
                            else:
                                continue
                        else:
                            break    
                    rand_num_list.append(rand_num)
                    output=self.df.iloc[rand_num,:-1].to_numpy()
                    for layer in self.net:
                        output=layer.forward(output)
                    output_iter_list.append(output)
                    start,end=rand_num,rand_num
                    end+=1
    
                output_iter_array=np.array(output_iter_list)
                
                #lossfunction
                error=self.obj_loss.lossFunction(self.df.iloc[start:end,-1].to_numpy(),output_iter_array)
                error_prim =self.obj_loss.lossFunction_prim(self.df.iloc[start:end,-1].to_numpy(),output_iter_array,self.df.iloc[start:end,:-1].to_numpy())
                
                #back propagation       
                grad_err_input=error_prim
                count=len(self.net)
                for back_layer in reversed(self.net):
                    if count%2!=0:
                        i=np.floor(count/2)
                        weight,bias,grad_err_input=back_layer.backward(self.LEARNING_RATE,grad_err_input,self.optimization)   
                        count-=1
                        result={'epoch':epoch,'iter':iter,'error':error,'error_prim':error_prim,'layer':i,'weight':weight,'bias':bias}
                        self.result_list.append(result)
                        result_temp.append(result)        
                    else:        
                        grad_err_input=back_layer.backward(self.LEARNING_RATE,grad_err_input,self.obj_opt)
                        count-=1
                
                result_tmp_df=pd.DataFrame(result_temp,columns=['epoch','iter','error','error_prim','layer','weight','bias']) 
                print(f'***********epoch={epoch}      error={result_tmp_df.error.mean()}***************')        

            
        self.result_df=pd.DataFrame(self.result_list,columns=['epoch','iter','error','error_prim','layer','weight','bias'])  
        self.stats=Stats().setParams(self.result_df)
        return self.result_df

    def errorToepoch(self):
        df=self.result_df.loc[self.result_df.layer==1.0]
        plt.plot(range(df.shape[0]),df.error)
        plt.show()
        
        
                   
                