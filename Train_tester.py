import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import Preprocessing as P
from tqdm import tqdm
import matplotlib.pyplot as plt

########### import model ############
# import xResNet as m
import Inception1d as m

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

train_loss = []
train_acc = []
val_loss = []
val_acc = []
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
        
        # calculate output accuracy
        pred = outputs.detach().numpy() > thresh_hold
        acc = (pred == labels.detach().numpy()).sum()/(inputs.shape[0]*5)
        
        # store loss
        running_loss += loss.item()
        train_loss.append(loss.item())
        train_acc.append(acc)
        print("epoch{}: loss per batch:{:.4f}".format(epoch+1, running_loss))
        print("Accuracy of epoch{} per batch:{}".format(epoch+1, acc*100))
    with torch.no_grad():
        net.eval()
        running_loss = 0.0
        correct_val = 0
    for inputs, labels in tqdm(val_data):
        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.float())
        # calculate output acc
        pred = outputs.detach().numpy() > thresh_hold
        acc = (pred == labels.detach().numpy()).sum()
        correct_val += acc
        #store loss
        running_loss += loss.item()
        val_loss.append(loss.item())
        val_acc.append(acc)
    print("========================================================")
    print("Validation loss of epoch{}:{:.4f}".format(epoch+1,running_loss))
    print("Accuracy of epoch{}:{}".format(epoch+1, correct_val*100/(y_val.shape[0]*5)))

print("\nfinished training =================================================")

#visualize
figure, axis = plt.subplots(2)
# training plot
axis[0].plot(train_loss,label="loss")
axis[0].plot(train_acc,label="acc")
axis[0].set_title("train data")
# validation plot
axis[1].plot(val_loss,label="loss")
axis[1].plot(val_acc,label="acc")
axis[1].set_title("validation data")

plt.legend()
plt.tight_layout()
plt.show()


