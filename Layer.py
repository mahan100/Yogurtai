from mimetypes import init
from msilib.schema import Class
from turtle import forward


class Layer():
    def __init__(self, input,output):
        self.input = None
        self.output = None
    
    def forward(self):
        pass
    def bakward(self):
        pass