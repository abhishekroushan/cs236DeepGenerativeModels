import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
print("imports done")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(1,1)

    def forward(self,x):
        #x=input
        x=self.fc1(x)
        return x

net=Net()
#net.cuda()
print(net)

inp=Variable(torch.randn(1,1,1), requires_grad=True)#tensor wrt which you can take grad
#print(inp)

import torch.optim as optim
#accuracy criteria
def criterion(out, label):
    return (label-out)**2
optimizer=optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

data=[(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

#training loop
for epoch in range(100):
    for i,data2 in enumerate(data):
        X,Y=iter(data2)
        X,Y=Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)#make it torch variable
        optimizer.zero_grad()
        outputs=net(X)#feed forward model
        loss=criterion(outputs,Y)
        loss.backward()
        optimizer.step()
        if (i%10==0):
            print("Epoch {}- loss={}".format(epoch, loss.data[0]))
        
