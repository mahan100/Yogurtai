import pandas as pd
from pandas_ml import ConfusionMatrix
import seaborn as sn
from matplotlib import pyplot as plt

class Metrics():
    def __init__(self,y_pred,y_true):
        pass

class ClassificationMetrics(Metrics):
    
    def __init__(self, y_pred, y_true):
        self.y_pred=y_pred
        self.y_true=y_true
        self.df=pd.DataFrame({'y_true':self.y_true,'y_pred':self.y_pred}
                             ,columns=['y_true','y_pred'])

    def confusionMatrix(self):
        confusion_matrix=pd.crosstab(self.df.y_true,self.df.y_pred,rownames=[self.df.y_true],colnames=[self.df.y_pred],margins=True)
        print(confusion_matrix)
        sn.heatmap(confusion_matrix,annot=True)
        plt.show()
    
    def classificationReport(self):
        confusion_matrix=ConfusionMatrix(self.df.y_true,self.df.y_pred)
        confusion_matrix.print_stats()
        
class RegresionMetrics(Metrics):
    pass