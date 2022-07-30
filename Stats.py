import numpy as np

class Stats():
    def __init__(self) -> None:
        pass
    def setParams(self,result_df):
        self.result=result_df
    def getParams(self):
        return self.result