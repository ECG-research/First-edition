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
import Inception1d as m
# import Modified_model1 as m
#hyperparameters
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 70
thresh_hold = 0.1
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#network
net = m.inception1d().float()
net.to(device)

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(net.parameters(), lr=1e-4)
# f1 = BinaryF1Score()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 6,eta_min=5e-5)
auc = average_precision_score

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
sle_train = torch.from_numpy(Sle[0])
sle_val = torch.from_numpy(Sle[1])
sle_test = torch.from_numpy(Sle[2])
#dataloader
train_data = DataLoader(TensorDataset(x_train,y_train,sle_train), batch_size=128, shuffle=True)

val_data = DataLoader(TensorDataset(x_val,y_val,sle_val), batch_size =128, shuffle=False)

test_data = DataLoader(TensorDataset(x_test,y_test,sle_test), batch_size =128, shuffle=False)

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

score_auc_all = []

# for t in range(3):
for epoch in tqdm(range(num_eporch)):
    net.train()
    print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
    if epoch%10 == 6:
        scheduler.step()
    train_acc = []
    train_loss = []
    train_auc = []
    train_score = []
    for inputs, labels, metadata in tqdm(train_data):
        inputs, labels, metadata = inputs.to(device),labels.to(device),metadata.to(device)
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
        train_loss.append(loss.item())
        train_auc.append(area)
        train_acc.append(acc)
        train_score.append(score)
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
        for inputs, labels, metadata in tqdm(val_data):
            inputs, labels, metadata = inputs.to(device),labels.to(device),metadata.to(device)

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


# figure, axis = plt.subplots(4)
# # training plot
# axis[0].plot(val_acc_all,label="val")
# axis[0].plot(train_acc_all,label="train")
# axis[0].set_title("accuracy")
# # validation plot
# axis[1].plot(val_loss_all,label="val")
# axis[1].plot(train_loss_all,label="train")
# axis[1].set_title("loss")

# axis[2].plot(val_auc_all,label="val")
# axis[2].plot(train_auc_all,label="train")
# axis[2].set_title("auc")

# axis[3].plot(val_score_all,label="val")
# axis[3].plot(train_score_all,label="train")
# axis[3].set_title("F1 score")

# plt.legend()
# plt.tight_layout()
# plt.savefig("Inception-5.jpg") #thay 12345 đẻ còn ra số khác nhau nha, em không muốn viết giấy đâu
# plt.show()


test_loss = []
test_acc = []
test_auc = []
with torch.no_grad():
    net.eval()
    batch_test = []
    for inputs, labels, metadata in tqdm(test_data):
        inputs, labels, metadata = inputs.to(device),labels.to(device),metadata.to(device)

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
        
        batch_test.append(acc)
        test_loss.append(loss.item())
        test_auc.append(area)
        test_acc.append(acc)
score1 = sum(test_auc)/len(test_auc)
score2 = max(val_auc_all)
score_auc_all.append((score1+score2)/2)
print("========================================================")
print("Test loss:{:.4f}".format(sum(test_loss)/len(test_loss)))    
print("Test accuracy:{:}".format(sum(batch_test)/len(batch_test)))
print("F1 score:{}".format(score))
print("auc:{}".format(score1))

print("Valmax_inception_score:{}".format(max(val_score_all)))
print("Valmax_inception_acc:{}".format(max(val_acc_all)))
print("Valmax_inception_auc:{}".format(score2))

print("total auc score:{}".format((score1+score2)/2))

# print(score_auc_all)
