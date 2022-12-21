import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import Preprocessing as P
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score

########### import model ############
# import xResNet as m
import dummy as m
# import Modified_model1 as m
#hyperparameters
path = ""
sampling_rate = 100
num_eporch = 75
thresh_hold = 0.5
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#network
net = m.inception1d().float()
# net = m.xresnet1d101_deeper().float() #could be other types of xresnet, see xResNet.py for more info
# net = m.modified_version().float()
net.to(device)

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(net.parameters(), lr=0.003)
# f1 = BinaryF1Score()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 6,eta_min=5e-5)
auc = average_precision_score

#data
data = P.Preprocessing(path,sampling_rate,experiment="diagnostic_superclass")
X = data.get_data_x()
x_train = np.concatenate((X[0][0],X[0][1],X[0][2]),axis = 1)
x_train = torch.from_numpy(x_train)
# x_train = [torch.from_numpy(X[0][i])for i in range(3)]
x_val = np.concatenate((X[1][0],X[1][1],X[1][2]),axis = 1)
x_val = torch.from_numpy(x_val)
# x_val = [torch.from_numpy(X[1][i])for i in range(3)]
x_test = np.concatenate((X[2][0],X[2][1],X[2][2]),axis = 1)
x_test = torch.from_numpy(x_test)
# x_test = [torch.from_numpy(X[2][i])for i in range(3)]

Y = data.get_data_y()
y_train = torch.from_numpy(Y[0])
y_val = torch.from_numpy(Y[1])
y_test = torch.from_numpy(Y[2])

Sle = data.get_data_metadata()
sle_train = torch.from_numpy(Sle[0])
sle_val = torch.from_numpy(Sle[1])
sle_test = torch.from_numpy(Sle[2])
#dataloader
# train_data = DataLoader(TensorDataset(x_train[0],x_train[1],x_train[2],sle_train,y_train), batch_size=200, shuffle=True)
# val_data = DataLoader(TensorDataset(x_val[0],x_val[1],x_val[2],sle_val,y_val), batch_size = 200, shuffle=False)
# test_data = DataLoader(TensorDataset(x_test[0],x_test[1],x_test[2],sle_test,y_test), batch_size = 200, shuffle=False)
train_data = DataLoader(TensorDataset(x_train,sle_train,y_train), batch_size=256, shuffle=True)

val_data = DataLoader(TensorDataset(x_val,sle_val,y_val), batch_size =256, shuffle=False)

test_data = DataLoader(TensorDataset(x_test,sle_test,y_test), batch_size =256, shuffle=False)

#train
train_loss = []
train_loss_all = []
train_acc = []
train_acc_all = []
train_auc = []
train_auc_all = []
train_score = []
train_score_all = []

val_loss = []
val_loss_all = []
val_acc = []
val_acc_all = []
val_auc = []
val_auc_all = []
val_score = []
val_score_all = []
for epoch in tqdm(range(num_eporch)):
    net.train()
    print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
    if epoch%10 == 6:
        scheduler.step()
    train_acc = []
    train_loss = []
    train_auc = []
    train_score = []
    for inputs,metadata,labels in tqdm(train_data):
        inputs,metadata,labels = inputs.to(device),metadata.to(device),labels.to(device)
        running_loss = 0.0
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float(),metadata.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        # calculate output accuracy and f1 score
        # score = f1(outputs,labels)
        pred = outputs.cpu().detach().numpy() > thresh_hold
        labels = labels.cpu().detach().numpy()
        score = f1_score(labels,pred,average='macro')
        
        area = auc(labels,outputs.cpu().detach().numpy())
        acc = (pred == labels).sum()/(inputs.shape[0]*5)
        
        # store loss
        running_loss += loss.item()
        train_loss.append(loss.item())
        train_auc.append(area)
        train_acc.append(acc)
        train_score.append(score)
        # print("epoch{}: loss per batch:{:.4f}".format(epoch+1, running_loss))
        # print("Accuracy of epoch{} per batch:{}".format(epoch+1, acc*100))
        # print("F1 score of epoch{}:{}".format(epoch+1,score))
    epoch_train_loss = sum(train_loss)/len(train_loss)
    epoch_train_acc = sum(train_acc)/len(train_acc)
    epoch_train_auc = sum(train_auc)/len(train_auc)
    epoch_train_score = sum(train_score)/len(train_score)
    train_loss_all.append(epoch_train_loss)
    train_acc_all.append(epoch_train_acc)
    train_auc_all.append(epoch_train_auc)
    train_score_all.append(epoch_train_score)
    print("epoch {}: Training loss:{:.4f}".format(epoch+1, epoch_train_loss))
    print("Training accuracy of epoch {}:{}".format(epoch+1, epoch_train_acc))
    print("Training F1 score of epoch {}:{}".format(epoch+1,epoch_train_score))
    print("Training AUC score of epoch {}:{}".format(epoch+1,epoch_train_auc))
    with torch.no_grad():
        val_loss = []
        val_acc = []
        val_auc = []
        val_score = []
        net.eval()
        running_loss = 0.0
        for inputs, metadata, labels in tqdm(val_data):
            inputs, metadata, labels = inputs.to(device),metadata.to(device), labels.to(device)
            # forward + backward + optimize
            outputs = net(inputs.float(),metadata.float())
            loss = criterion(outputs, labels.float())
            # calculate output acc
            # score = f1(outputs,labels)
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()
            score = f1_score(labels,pred,average='macro')
            area = auc(labels,outputs.cpu().detach().numpy())
            acc = (pred == labels).sum()/(inputs.shape[0]*5)
            #store loss
            running_loss += loss.item()
            val_loss.append(loss.item())
            val_auc.append(area)
            val_acc.append(acc)
            val_score.append(score)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        epoch_val_acc = sum(val_acc)/len(val_acc)
        epoch_val_auc = sum(val_auc)/len(val_auc)
        epoch_val_score = sum(val_score)/len(val_score)
        val_loss_all.append(epoch_val_loss)
        val_acc_all.append(epoch_val_acc)
        val_auc_all.append(epoch_val_auc)
        val_score_all.append(epoch_val_score)
        print("========================================================")
        print("Validation loss of epoch {}:{:.4f}".format(epoch+1,epoch_val_loss))
        print("Validation accuracy of epoch {}:{}".format(epoch+1, epoch_val_acc))
        print("Validation F1 score of epoch {}:{}".format(epoch+1, epoch_val_score))
        print("Validation AUC score of epoch {}:{}".format(epoch+1, epoch_val_auc))


print("\nfinished training =================================================")


figure, axis = plt.subplots(4)
# training plot
axis[0].plot(val_acc_all,label="val")
axis[0].plot(train_acc_all,label="train")
axis[0].set_title("accuracy")
# validation plot
axis[1].plot(val_loss_all,label="val")
axis[1].plot(train_loss_all,label="train")
axis[1].set_title("loss")

axis[2].plot(val_auc_all,label="val")
axis[2].plot(train_auc_all,label="train")
axis[2].set_title("auc")

axis[3].plot(val_score_all,label="val")
axis[3].plot(train_score_all,label="train")
axis[3].set_title("F1 score")

plt.legend()
plt.tight_layout()
plt.savefig("Inception-2.jpg") #thay 12345 đẻ còn ra số khác nhau nha, em không muốn viết giấy đâu
plt.show()


test_loss = []
test_acc = []
test_auc = []
with torch.no_grad():
    net.eval()
    running_loss = 0.0
    batch_test = []
    for inputs, metadata, labels in tqdm(test_data):
        inputs,metadata,labels = inputs.to(device),metadata.to(device),labels.to(device)
        outputs = net(inputs.float(),metadata.float())
        loss = criterion(outputs, labels.float())
        # calculate output acc
        # score = f1(outputs,labels)
        pred = outputs.cpu().detach().numpy() > thresh_hold
        labels = labels.cpu().detach().numpy()
        score = f1_score(labels,pred,average='macro')
        area = auc(labels,outputs.cpu().detach().numpy())
        acc = (pred == labels).sum()/(inputs.shape[0]*5)
        
        batch_test.append(acc)
        running_loss += loss.item()
        test_loss.append(loss.item())
        test_auc.append(area)
        test_acc.append(acc)
print("========================================================")
print("Test loss:{:.4f}".format(sum(test_loss)/len(test_loss)))    
print("Test accuracy:{:}".format(sum(batch_test)/len(batch_test)))
print("F1 score:{}".format(score))
print("Test_auc:{}".format(test_auc[-1]))

print("Val_inception_score:{}".format(val_score_all[-1]))
print("Val_inception_acc:{}".format(val_acc_all[-1]))
print("Val_inception_auc:{}".format(val_auc_all[-1]))

print("Total_auc: {}".format((val_auc_all[-1]+test_auc[-1])/2))

#ờ nó đây
