import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# import xResNet as m
import Inception1d as m
import Preprocessing as P
from tqdm import tqdm

#hyperparameters
path = ""
sampling_rate = 100
num_eporch = 10
thresh_hold = 0.5

#network
net = m.inception1d().float()
# net = m.xresnet1d50_deeper().float() #could be other types of xresnet, see xResNet.py for more info

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(net.parameters(), lr=0.001)

#data
data = P.Preprocessing(path,sampling_rate,experiment="diagnostic_superclass")

X = data.get_data_x()
x_train = np.concatenate((X[0][0],X[0][1],X[0][2]),axis = 1)
x_train = torch.from_numpy(x_train)
x_val = np.concatenate((X[1][0],X[1][1],X[1][2]),axis = 1)
x_val = torch.from_numpy(x_val)
x_test = np.concatenate((X[2][0],X[2][1],X[2][2]),axis = 1)
x_test = torch.from_numpy(x_test)

Y = data.get_data_y()
y_train = torch.from_numpy(Y[0])
y_val = torch.from_numpy(Y[1])
y_test = torch.from_numpy(Y[2])

Sle = data.get_data_metadata()

#dataloader
train_data = DataLoader(TensorDataset(x_train,y_train), batch_size=200, shuffle=True)
val_data = DataLoader(TensorDataset(x_val,y_val), batch_size = 200, shuffle=False)
test_data = DataLoader(TensorDataset(x_test,y_test), batch_size = 200, shuffle=False)

#train
for epoch in tqdm(range(num_eporch)):
    net.train()
    print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
    
    for inputs, labels in tqdm(train_data):
        running_loss = 0.0
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        pred = outputs.detach().numpy() > thresh_hold
        
        running_loss += loss.item()
        print("epoch{}: loss per batch:{:.4f}".format(epoch+1, running_loss))
        print("Accuracy of epoch{} per batch:{}".format(epoch+1, (pred == labels.detach().numpy()).sum()*100/(inputs.shape[0]*5)))
    with torch.no_grad():
        net.eval()
        running_loss = 0.0
        correct_val = 0
    for inputs, labels in tqdm(val_data):
        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.float())
        
        running_loss += loss.item()
        pred = outputs.detach().numpy() > thresh_hold
        correct_val += (pred == labels.detach().numpy()).sum()
        
    print("Accuracy of epoch{}:{:}".format(epoch+1, correct_val*100/(y_val.shape[0]*5)))

print("finished")
    