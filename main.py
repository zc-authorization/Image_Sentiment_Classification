from torch.utils.data import DataLoader
import pandas as pd
from ConvNet import ConvNet
import torch
import torchvision.transforms as transforms
from ISCDataset import ISCDataset,ToTensor
#加载训练数据
sentiment_dataset=ISCDataset(csv_file='data/train.csv',transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(sentiment_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

net=ConvNet()
#print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.1)
loss_fn=torch.nn.CrossEntropyLoss()

for i in range(300):
    num_verificationData=0
    num_right=0
    for i_batch,item in enumerate(trainloader):
        if i_batch<=6000:
            #print(type(item['image']),item['image'].shape)
            prediction=net(item['image'])
            ##print(prediction)
            loss=loss_fn(prediction,item['lable'])

            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            #print(len(item['lable']))
            with torch.no_grad():
                num_verificationData+=len(item['lable'])
                veri_prediction=net(item['image'])
                #print(item['image'])
                ##print(veri_prediction)
                veri_prediction=torch.argmax(veri_prediction,dim=1).numpy()
                #print(veri_prediction,item['lable'])
                for j in range(len(item['lable'])):
                    if veri_prediction[j]==item['lable'].numpy()[j]:
                        num_right+=1
                ##print(num_right)

    print('第 %d 次训练的正确率为： %.4f %%'%(i+1,num_right/num_verificationData*100))



 #读入test.csv
test_dataset=ISCDataset('data/test.csv',transform=transforms.Compose([ToTensor()]))
testloader=DataLoader(test_dataset,batch_size=4,shuffle=False)
 #读入sample.csv
sample=pd.read_csv('data/sample.csv')
for i_batch,item in enumerate(testloader):
    prediction=net(item['image'])
    prediction=torch.argmax(prediction,dim=1).numpy()
    for i in range(len(prediction)):
        sample.iloc[item['lable'].numpy()[i],1]=prediction[i]
#sample.iloc[0,1]=10
#print(sample,type(sample))
sample.to_csv('data/sample.csv',index=False)



