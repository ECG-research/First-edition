import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# import Preprocessing as P
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score


########### import model ############
import ST_CNN as model
net = model.ST_CNN5(num_class=5).float()

#hyperparameters
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 20
thresh_hold = 0.5
batch_size = 128
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#network
net.to(device)

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(net.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 3)

#data
ytrain = np.load("y_super_train.npy")
yval = np.load("y_super_val.npy")
ytest = np.load("y_super_test.npy")
# ytrain = np.load("y_sub_train.npy")
# yval = np.load("y_sub_val.npy")
# ytest = np.load("y_sub_test.npy")
ytrain = torch.from_numpy(ytrain)
yval = torch.from_numpy(yval)
ytest = torch.from_numpy(ytest)

xtrain = np.load("X_train_bandpass.npy")
xval = np.load("X_val_bandpass.npy")
xtest = np.load("X_test_bandpass.npy")
xtrain = np.expand_dims(xtrain,3)
xval = np.expand_dims(xval,3)
xtest = np.expand_dims(xtest,3)
xtrain = torch.from_numpy(xtrain)
xval = torch.from_numpy(xval)
xtest = torch.from_numpy(xtest)


# dataloader
train_data = DataLoader(TensorDataset(xtrain,ytrain), batch_size=batch_size, shuffle=True)
val_data = DataLoader(TensorDataset(xval,yval), batch_size=batch_size, shuffle=False)
test_data = DataLoader(TensorDataset(xtest,ytest), batch_size=batch_size, shuffle=False)

#train

val_auc_all = []

count = 0
max_count = 3
for t in range(1):
    for epoch in tqdm(range(num_eporch)):
        net.train()
        print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
        all_outs_train = []
        all_labels_train = []
        all_loss_train = []
        for inputs,labels in tqdm(train_data):
            inputs,labels = inputs.to(device).float(),labels.to(device).float()
            running_loss = 0.0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss += 0.005*torch.norm(net.dense1.dense[0].weight, p = 2) + 0.009*torch.norm(net.dense2.dense[0].weight, p = 2)
            loss.backward()
            optimizer.step()

            all_outs_train.append(outputs.cpu().detach().numpy())
            all_labels_train.append(labels.cpu().detach().numpy())
            all_loss_train.append(loss)
        all_outs_train = np.concatenate(all_outs_train,axis = 0)
        all_labels_train = np.concatenate(all_labels_train,axis = 0)
        all_preds_train = all_outs_train > thresh_hold
        print("epoch {}: Training loss:{:.4f}".format(epoch+1, sum(all_loss_train)/len(all_loss_train)))
        print("Training F1 score of epoch {}:{}".format(epoch+1,f1_score(all_labels_train,all_preds_train,average='macro')))
        print("Training AUC score of epoch {}:{}".format(epoch+1,roc_auc_score(all_labels_train,all_outs_train,average = "macro")))

        with torch.no_grad():
            net.eval()
            running_loss = 0.0
            all_outs_val = []
            all_labels_val = []
            all_loss_val = []
            for inputs,labels in tqdm(val_data):
                inputs,labels = inputs.to(device).float(),labels.to(device).float()
                running_loss = 0.0
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.float())
                loss += 0.005*torch.norm(net.dense1.dense[0].weight, p = 2) + 0.009*torch.norm(net.dense2.dense[0].weight, p = 2)

                all_outs_val.append(outputs.cpu().detach().numpy())
                all_labels_val.append(labels.cpu().detach().numpy())
                all_loss_val.append(loss)
            all_outs_val = np.concatenate(all_outs_val,axis = 0)
            all_labels_val = np.concatenate(all_labels_val,axis = 0)
            all_preds_val = all_outs_val > thresh_hold
            val_f1 = f1_score(all_labels_val,all_preds_val,average='macro')
            val_auc_all.append(roc_auc_score(all_labels_val,all_outs_val,average = "macro"))
            # epoch_val_score = sum(val_score)/len(val_score)
            if epoch >= 20:    
                if val_auc_all[-1] < val_auc_all[-2]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
            
            print("epoch {}: Validation loss:{:.4f}".format(epoch+1, sum(all_loss_val)/len(all_loss_val)))
            print("Validation F1 score of epoch {}:{}".format(epoch+1,val_f1))
            print("Validation AUC score of epoch {}:{}".format(epoch+1,val_auc_all[-1]))


    print("\nfinished training =================================================")

    test_f1 = 0
    test_auc = 0
    with torch.no_grad():
        net.eval()
        running_loss = 0.0
        all_outs_test = []
        all_labels_test = []
        all_loss_test = []
        for inputs,labels in tqdm(test_data):
            inputs,labels = inputs.to(device).float(),labels.to(device).float()
            running_loss = 0.0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss += 0.005*torch.norm(net.dense1.dense[0].weight, p = 2) + 0.009*torch.norm(net.dense2.dense[0].weight, p = 2)
            
            # calculate output acc
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()

            all_outs_test.append(outputs.cpu().detach().numpy())
            all_labels_test.append(labels)
            all_loss_test.append(loss)
        all_outs_test = np.concatenate(all_outs_test,axis = 0)
        all_labels_test = np.concatenate(all_labels_test,axis = 0)
        all_preds_test = all_outs_test > thresh_hold
        test_f1 = f1_score(all_labels_test,all_preds_test,average='macro')
        test_auc = roc_auc_score(all_labels_test,all_outs_test,average = "macro")
        print("epoch {}: Test loss:{:.4f}".format(epoch+1, sum(all_loss_test)/len(all_loss_test)))
        print("Test F1 score of epoch {}:{}".format(epoch+1,test_f1))
        print("Test AUC score of epoch {}:{}".format(epoch+1,test_auc))

print("\nfinished running =============================================================================")
print("Val max auc:{}".format(max(val_auc_all)))
print("Test auc:{}".format(test_auc))
print("the scores were:{}".format((max(val_auc_all)+test_auc)/2))

# torch.save(net.pm.state_dict(),"dwt_pm_test")
# torch.save(net.pm.state_dict(),"raw_pm_test_4")
# torch.save(net.classifier.state_dict(),"pseudo_test1")
