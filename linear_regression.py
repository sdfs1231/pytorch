#linear regression

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-5,5,100),dim=1)
y = x.pow(2)+5*torch.rand(x.size())

x = Variable(x)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hiddens,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens,n_output)


    def forward(self,x):
        x = F.relu(self.hidden(x))
        pre_y = self.predict(x)
        return pre_y

net = Net(1,10,1)
# print(net)
plt.ion()
plt.show()
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.MSELoss()

for t in range(10000):
    predictions = net(x)

    loss = loss_func(predictions,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%100 ==0:
        print(loss)
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),predictions.data.numpy(),'r-',lw=3)
        plt.text(0.5,0,'Loss=%.4f'%loss.item())
        plt.pause(0.1)
plt.ioff()
plt.show()