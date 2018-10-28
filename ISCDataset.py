import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms,datasets


class ISCDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.data=pd.read_csv(csv_file)
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image=self.data.iloc[idx,1].split()
        image=list(map(float,image))
        image=np.array(image).reshape(-1,48)
        lable=np.array(self.data.iloc[idx,0])

        sample={'image':image,'lable':lable}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, lable = sample['image'], sample['lable']
        return {'image': torch.unsqueeze(torch.from_numpy(image),0),
                'lable': torch.from_numpy(lable)}



# sen_dataset=ISCDataset(csv_file='data/train.csv',transform=transforms.Compose([ToTensor()]))
#
# sample=sen_dataset[2]
# #print(sample['image'],sample['lable'])
#
# dataloader = DataLoader(sen_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)
#
#
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['lable'].size())
