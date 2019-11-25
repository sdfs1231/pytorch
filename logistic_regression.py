#linear regression

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

x ,y= Variable(x),Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
# exit()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hiddens,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens,n_output)


    def forward(self,x):
        x = F.relu(self.hidden(x))
        pre_y = self.predict(x)
        return pre_y

net = Net(2,10,2)
# print(net)
plt.ion()
plt.show()
optimizer = torch.optim.SGD(net.parameters(),lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(10000):
    out = net(x)

    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%100 ==0:
        plt.cla()
        _,predictions = torch.max(F.softmax(out),1)
        pre_y = predictions.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pre_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pre_y==target_y)/200

        plt.text(1.5,-4,'accurace=%.4f'%accuracy)
        plt.show()
        plt.pause(0.1)
plt.ioff()
