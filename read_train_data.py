import pandas as pd
import numpy as np

def read_trainData():
    data=pd.read_csv('data/train.csv')
    print(type(data.iloc[0,0]))
    print(len(data))
    print(data.iloc[0,1])
    data=data.iloc[0,1].split()
    #num=data.as_matrix()
    data=list(map(float,data))
    data=np.array(data)
    num=data.reshape(-1,48)


    print(num.shape)
    print(num[:4])



read_trainData()
