import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ISCDataset import ISCDataset,ToTensor
import torch
import torchvision.transforms as transforms

def update_lable():
    #读入test.csv
    # test_dataset=ISCDataset('data/test.csv',transform=transforms.Compose([ToTensor()]))
    # testloader=DataLoader(test_dataset,batch_size=4,shuffle=False)
    #读入sample.csv
    sample=pd.read_csv('data/sample.csv')
    sample.iloc[0,1]=10
    print(sample,type(sample))
    sample.to_csv('data/sample.csv',index=False)

update_lable()
