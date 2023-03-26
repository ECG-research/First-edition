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

raw = True
pseudo = False

if raw:
    import second_edition as model
    if pseudo:
        p_classifier = model.Pseudo_classifier(num_class=23)
        p_classifier.load_state_dict(torch.load("pseudo"))
        p_classifier.eval()
        for param in p_classifier.parameters():
            param.requires_grad = False
        net = model.Branch(type = "raw",classifier=p_classifier, num_class=23).float()
    else:
        net = model.Branch(type = "raw",num_class=23).float()
else:
    import Inception1d as model
    net = model.inception1d().float()

def input_incep(inputs,metadata):
    if raw:
        return net(inputs.float(),metadata.float())
    else:
        return net(inputs.float())

# import ST_CNN as model
# net = model.ST_CNN5(num_class=5).float()

#hyperparameter
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 100
thresh_hold = 0.5
batch_size = 128
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#network
net.to(device)
def load_pm(path, type):
    pm = model.Projection_Module(type = type)
    pm.load_state_dict(torch.load(path))
    pm.eval()
    for param in pm.parameters():
        param.requires_grad = False
    return pm

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy
optimizer = optim.AdamW(net.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 4,eta_min=5e-6)

#data
# ytrain = np.load("y_super_train.npy")
# yval = np.load("y_super_val.npy")
# ytest = np.load("y_super_test.npy")
ytrain = np.load("y_sub_train.npy")
yval = np.load("y_sub_val.npy")
ytest = np.load("y_sub_test.npy")
ytrain = torch.from_numpy(ytrain)
yval = torch.from_numpy(yval)
ytest = torch.from_numpy(ytest)

sletrain = np.load("sletrain.npy")
sleval = np.load("sleval.npy")
sletest = np.load("sletest.npy")
sletrain = torch.from_numpy(sletrain)
sleval = torch.from_numpy(sleval)
sletest = torch.from_numpy(sletest)


dwt_train = np.load("dwt_train.npy")
dwt_val = np.load("dwt_val.npy")
dwt_test = np.load("dwt_test.npy")
# dwt_train = np.expand_dims(dwt_train,3)
# dwt_val = np.expand_dims(dwt_val,3)
# dwt_test = np.expand_dims(dwt_test,3)
dwt_train = torch.from_numpy(dwt_train)
dwt_val = torch.from_numpy(dwt_val)
dwt_test = torch.from_numpy(dwt_test)


# dataloader
train_data = DataLoader(TensorDataset(dwt_train,sletrain,ytrain), batch_size=batch_size, shuffle=True)
val_data = DataLoader(TensorDataset(dwt_val,sleval,yval), batch_size=batch_size, shuffle=False)
test_data = DataLoader(TensorDataset(dwt_test,sletest,ytest), batch_size=batch_size, shuffle=False)

#train

val_auc_all = []

count = 0
max_count = 2
for t in range(1):
    for epoch in tqdm(range(num_eporch)):
        net.train()
        print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
        if epoch%4 == 3 and epoch < 16:
            scheduler.step()
        all_outs_train = []
        all_labels_train = []
        all_loss_train = []
        for inputs,metadata,labels in tqdm(train_data):
            inputs,metadata,labels = inputs.to(device).float(),metadata.to(device).float(),labels.to(device).float()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = input_incep(inputs,metadata)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            all_outs_train.append(outputs.cpu().detach().numpy())
            all_labels_train.append(labels.cpu().detach().numpy())
            all_loss_train.append(loss)
        all_outs_train = np.concatenate(all_outs_train,axis = 0)
        all_labels_train = np.concatenate(all_labels_train,axis = 0)
        f1 = 0
        thresh_hold = 0
        for i in range(10):
            thresh_hold += 0.1
            all_preds_train = all_outs_train > thresh_hold
            f_test = f1_score(all_labels_train,all_preds_train,average='macro')
            if f1 < f_test:
                f1 = f_test
        print("epoch {}: Training loss:{:.4f}".format(epoch+1, sum(all_loss_train)/len(all_loss_train)))
        # print("Training F1 score of epoch {}:{}".format(epoch+1,f1_score(all_labels_train,all_preds_train,average='macro')))
        print("Training F1 score of epoch {}:{}".format(epoch+1,f1))
        print("Training AUC score of epoch {}:{}".format(epoch+1,roc_auc_score(all_labels_train,all_outs_train,average = "macro")))

        with torch.no_grad():
            net.eval()
            all_outs_val = []
            all_labels_val = []
            all_loss_val = []
            for inputs,metadata,labels in tqdm(val_data):
                inputs,metadata,labels = inputs.to(device).float(),metadata.to(device).float(),labels.to(device).float()
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = input_incep(inputs,metadata)
                loss = criterion(outputs, labels.float())

                all_outs_val.append(outputs.cpu().detach().numpy())
                all_labels_val.append(labels.cpu().detach().numpy())
                all_loss_val.append(loss)
            all_outs_val = np.concatenate(all_outs_val,axis = 0)
            all_labels_val = np.concatenate(all_labels_val,axis = 0)
            f1 = 0
            thresh_hold = 0
            for i in range(10):
                thresh_hold += 0.1
                all_preds_val = all_outs_val > thresh_hold
                f_test = f1_score(all_labels_val,all_preds_val,average='macro')
                if f1 < f_test:
                    f1 = f_test

            val_auc_all.append(roc_auc_score(all_labels_val,all_outs_val,average = "macro"))
            if epoch >= 16:    
                if val_auc_all[-1] < val_auc_all[-2]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
            
            print("epoch {}: Validation loss:{:.4f}".format(epoch+1, sum(all_loss_val)/len(all_loss_val)))
            # print("Validation F1 score of epoch {}:{}".format(epoch+1,val_f1))
            print("Validation F1 score of epoch {}:{}".format(epoch+1,f1))
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
        for inputs,metadata,labels in tqdm(test_data):
            inputs,metadata,labels = inputs.to(device).float(),metadata.to(device).float(),labels.to(device).float()
            running_loss = 0.0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = input_incep(inputs,metadata)
            loss = criterion(outputs, labels.float())
            
            # calculate output acc
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()

            all_outs_test.append(outputs.cpu().detach().numpy())
            all_labels_test.append(labels)
            all_loss_test.append(loss)
        all_outs_test = np.concatenate(all_outs_test,axis = 0)
        all_labels_test = np.concatenate(all_labels_test,axis = 0)
        f1 = 0
        thresh_hold = 0
        for i in range(10):
            thresh_hold += 0.1
            all_preds_test = all_outs_test > thresh_hold
            f_test = f1_score(all_labels_test,all_preds_test,average='macro')
            if f1 < f_test:
                f1 = f_test
        test_auc = roc_auc_score(all_labels_test,all_outs_test,average = "macro")
        print("epoch {}: Test loss:{:.4f}".format(epoch+1, sum(all_loss_test)/len(all_loss_test)))
        print("Test F1 score of epoch {}:{}".format(epoch+1,f1))
        print("Test AUC score of epoch {}:{}".format(epoch+1,test_auc))

print("\nfinished running =============================================================================")
print("Val max auc:{}".format(max(val_auc_all)))
print("Test auc:{}".format(test_auc))
print("the scores were:{}".format((max(val_auc_all)+test_auc)/2))

torch.save(net.pm.state_dict(),"dwt_pm")
# torch.save(net.pm.state_dict(),"raw_pm")
# torch.save(net.classifier.state_dict(),"pseudo")