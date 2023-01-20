import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import Preprocessing as P
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score
import second_edition as model

sampling_rate = 100
num_eporch = 1
thresh_hold = 0.5

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

# branch network
p_classifier = model.Pseudo_classifier()
p_classifier.load_state_dict(torch.load("pseudo_0694"))
p_classifier.eval()
for param in p_classifier.parameters():
    param.requires_grad = False

branch_wl = model.Branch(type="wavelet", classifier=p_classifier).to(device)

# evaluation methods
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(branch_wl.parameters(), lr=1e-4)
# f1 = BinaryF1Score()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 5,eta_min=5e-6)
auc = average_precision_score

# xtrain = np.load("X_train.npy")
# xval = np.load("X_val.npy")
# xtest = np.load("X_test.npy")

ytrain = np.load("ytrain.npy")
yval = np.load("yval.npy")
ytest = np.load("ytest.npy")
ytrain = torch.from_numpy(ytrain)
yval = torch.from_numpy(yval)
ytest = torch.from_numpy(ytest)

sletrain = np.load("sletrain.npy")
sleval = np.load("sleval.npy")
sletest = np.load("sletest.npy")
sletrain = torch.from_numpy(sletrain)
sleval = torch.from_numpy(sleval)
sletest = torch.from_numpy(sletest)


cwt_train = np.load("cwt_train.npy")
cwt_val = np.load("cwt_val.npy")
cwt_test = np.load("cwt_test.npy")
cwt_train = torch.from_numpy(cwt_train)
cwt_val = torch.from_numpy(cwt_val)
cwt_test = torch.from_numpy(cwt_test)

# dataloader
train_data = DataLoader(TensorDataset(cwt_train,sletrain,ytrain), batch_size=1, shuffle=True)
val_data = DataLoader(TensorDataset(cwt_val,sleval,yval), batch_size = 1, shuffle=False)
test_data = DataLoader(TensorDataset(cwt_test,sletest,ytest), batch_size = 1, shuffle=False)

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

auc_all = []
loss_all = []

count = 0
max_count = 2
for t in range(1):
    for epoch in tqdm(range(num_eporch)):
        branch_wl.train()
        print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
        if epoch%3 == 2:
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
            outputs = branch_wl(inputs.float(),metadata.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            # calculate output accuracy and f1 score
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
            branch_wl.eval()
            running_loss = 0.0
            for inputs, metadata, labels in tqdm(val_data):
                inputs, metadata, labels = inputs.to(device),metadata.to(device), labels.to(device)
                # forward + backward + optimize
                outputs = branch_wl(inputs.float(),metadata.float())
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
            if epoch >= 2:    
                if epoch_val_auc < val_auc_all[-1]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
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

    test_loss = []
    test_score = []
    test_acc = []
    test_auc = []
    with torch.no_grad():
        branch_wl.eval()
        running_loss = 0.0
        batch_test = []
        for inputs, metadata, labels in tqdm(test_data):
            inputs,metadata,labels = inputs.to(device),metadata.to(device),labels.to(device)
            outputs = branch_wl(inputs.float(),metadata.float())
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
            test_score.append(score)
    score1 = sum(test_auc)/len(test_auc)
    score2 = max(val_auc_all)
    losses = sum(test_loss)/len(test_loss)
    print("========================================================")
    print("Test loss:{:.4f}".format(losses))    
    print("Test accuracy:{:}".format(sum(batch_test)/len(batch_test)))
    print("F1 score:{}".format(sum(test_score)/len(test_score)))
    print("Test_auc:{}".format(score1))
    print("Val_inception_auc:{}".format(score2))
    auc_all.append(score1)
    auc_all.append(score2)
    loss_all.append(losses)

print("\nfinished running =============================================================================")
print("the scores were:")
print(auc_all)
print(sum(auc_all)/len(auc_all))
print(loss_all)

torch.save(branch_wl.pm.state_dict(),"wl_pm")
# pm = model.Projection_Module()
# pm.load_state_dict(torch.load("stft_pm"))
# pm.eval()
# print(pm(stft_val.float(),sleval.float()))