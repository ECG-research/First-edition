import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import Preprocessing as P
from tqdm import tqdm
import matplotlib.pyplot as plt
from Metrics import challenge_metrics, f1
from sklearn.metrics import f1_score, average_precision_score

########### import model ############
# import xResNet as m
# import Inception1d as m
import Modified_model1 as m
#hyperparameters
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 50
thresh_hold = 0.1

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

#network
# net = m.inception1d().float()
# net = m.xresnet1d101_deeper().float() #could be other types of xresnet, see xResNet.py for more info
net = m.modified_version().float().to(device)

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(net.parameters(), lr=0.005)
# f1 = BinaryF1Score()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 4,eta_min=1e-4)
auc = average_precision_score

#data
data = P.Preprocessing(path,sampling_rate,experiment="diagnostic_superclass")
X = data.get_data_x()
# x_train = np.concatenate((X[0][0],X[0][1],X[0][2]),axis = 1)
x_train = [torch.from_numpy(X[0][i])for i in range(3)]
# x_val = np.concatenate((X[1][0],X[1][1],X[1][2]),axis = 1)
x_val = [torch.from_numpy(X[1][i])for i in range(3)]
# x_test = np.concatenate((X[2][0],X[2][1],X[2][2]),axis = 1)
x_test = [torch.from_numpy(X[2][i])for i in range(3)]

Y = data.get_data_y()
y_train = torch.from_numpy(Y[0])
y_val = torch.from_numpy(Y[1])
y_test = torch.from_numpy(Y[2])

Sle = data.get_data_metadata()
sle_train = torch.from_numpy(Sle[0])
sle_val = torch.from_numpy(Sle[1])
sle_test = torch.from_numpy(Sle[2])
#dataloader
train_data = DataLoader(TensorDataset(x_train[0],x_train[1],x_train[2],sle_train,y_train), batch_size=200, shuffle=True)
val_data = DataLoader(TensorDataset(x_val[0],x_val[1],x_val[2],sle_val,y_val), batch_size = 200, shuffle=False)
test_data = DataLoader(TensorDataset(x_test[0],x_test[1],x_test[2],sle_test,y_test), batch_size = 200, shuffle=False)


#train
train_loss = []
train_acc = []
train_auc = []
val_loss = []
val_acc = []
val_auc = []
val_score = []
val_score_all = []
for epoch in tqdm(range(num_eporch)):
    net.train()
    print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
    if epoch%10 == 6:
            scheduler.step()
    
    for input1, input2, input3, metadata, labels in tqdm(train_data):
        running_loss = 0.0
        optimizer.zero_grad()
        input1, input2, input3, metadata, labels = input1.to(device), input2.to(device), input3.to(device), metadata.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = net(input1.float(),input2.float(),input3.float(),metadata.float())
        outputs.to(device)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        # calculate output accuracy and f1 score
        # score = f1(outputs,labels)
        pred = outputs.cpu().detach().numpy() > thresh_hold
        labels = labels.cpu().detach().numpy()
        score = f1_score(labels,pred,average='macro')
        area = auc(labels,pred)
        acc = (pred == labels).sum()/(input1.shape[0]*5)
        
        # store loss
        running_loss += loss.item()
        train_loss.append(loss.item())
        train_auc.append(area)
        train_acc.append(acc)
        print("epoch{}: loss per batch:{:.4f}".format(epoch+1, running_loss))
        print("Accuracy of epoch{} per batch:{}".format(epoch+1, acc*100))
        print("F1 score of epoch{}:{}".format(epoch+1,score))
    with torch.no_grad():
        net.eval()
        running_loss = 0.0
        for input1, input2, input3, metadata, labels in tqdm(val_data):
            input1, input2, input3, metadata, labels = input1.to(device), input2.to(device), input3.to(device), metadata.to(device), labels.to(device)
            # forward + backward + optimize
            outputs = net(input1.float(),input2.float(),input3.float(),metadata.float())            
            outputs.to(device)
            loss = criterion(outputs, labels.float())
            # calculate output acc
            # score = f1(outputs,labels)
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()
            score = f1_score(labels,pred,average='macro')
            area = auc(labels,pred)
            acc = (pred == labels).sum()/(input1.shape[0]*5)
            #store loss
            running_loss += loss.item()
            val_loss.append(loss.item())
            val_auc.append(area)
            val_acc.append(acc)
            val_score.append(score)
        t = sum(val_score)/len(val_score)
        print("========================================================")
        print("Validation loss of epoch{}:{:.4f}".format(epoch+1,sum(val_loss)/len(val_loss)))
        print("Accuracy of epoch{}:{}".format(epoch+1, sum(val_acc)/len(val_acc)*100))
        print("F1 score of epoch{}:{}".format(epoch+1,t))
        val_score_all.append(t)


print("\nfinished training =================================================")


figure, axis = plt.subplots(3)
# training plot
axis[0].plot(val_acc,label="val")
axis[0].plot(train_acc,label="train")
axis[0].set_title("accuracy")
# validation plot
axis[1].plot(val_loss,label="val")
axis[1].plot(train_loss,label="train")
axis[1].set_title("loss")

axis[2].plot(val_auc,label="val")
axis[2].plot(train_auc,label="train")
# axis[2].set_title("auc")

plt.legend()
plt.tight_layout()
plt.show()


test_loss = []
test_acc = []
test_auc = []
with torch.no_grad():
    net.eval()
    running_loss = 0.0
    batch_test = []
    for input1, input2, input3, metadata, labels in tqdm(val_data):
        input1, input2, input3, metadata, labels = input1.to(device), input2.to(device), input3.to(device), metadata.to(device), labels.to(device)
        # forward + backward + optimize
        outputs = net(input1.float(),input2.float(),input3.float(),metadata.float()) 
        outputs.to(device)
        loss = criterion(outputs, labels.float())
        # calculate output acc
        # score = f1(outputs,labels)
        pred = outputs.cpu().detach().numpy() > thresh_hold
        labels = labels.cpu().detach().numpy()
        score = f1_score(labels,pred,average='macro')
        area = auc(labels,pred)
        acc = (pred == labels).sum()/(input1.shape[0]*5)
        
        batch_test.append(acc)
        running_loss += loss.item()
        test_loss.append(loss.item())
        test_auc.append(area)
        test_acc.append(acc)
print("========================================================")
print("Test loss:{:.4f}".format(sum(test_loss)/len(test_loss)))    
print("Test accuracy:{:}".format(sum(batch_test)/len(batch_test)*100))
print("F1 score:{}".format(score))
print("VAL_score:{}".format(val_score_all))