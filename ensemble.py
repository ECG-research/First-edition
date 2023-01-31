import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import second_edition as model

# transfering pseudoclassifier
# p_classifier = model.Pseudo_classifier()
# p_classifier.load_state_dict(torch.load("pseudo_test"))
# p_classifier.eval()
# for param in p_classifier.parameters():
#     param.requires_grad = False

def load_pm(path, type):
    pm = model.Projection_Module(type = type)
    pm.load_state_dict(torch.load(path))
    pm.eval()
    for param in pm.parameters():
        param.requires_grad = False
    return pm

#hyperparameters
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 100
thresh_hold = 0.5
batch_size = 128
e2e = False
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#load projection module
if not e2e:
    pm_raw = load_pm("raw_pm_test","raw")
    pm_dwt = load_pm("dwt_pm_test","raw") # type is raw because I used inception for dwt
    pm_stft = load_pm("stft_pm_test","stft")
else:
    pm_raw = model.Projection_Module(type = "raw")
    pm_dwt = model.Projection_Module(type = "raw")
    pm_stft = model.Projection_Module(type = "stft")
pms = [pm_raw,pm_dwt,pm_stft]
# initialize network
net = model.Second_edition(pms).to(device)

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits
distance = nn.functional.cosine_similarity

def criterion_func(output,label,pred1,pred2,pred3,alpha=0.1):
    pred1 = torch.squeeze(pred1,dim = 1)
    pred2 = torch.squeeze(pred2,dim = 1)
    pred3 = torch.squeeze(pred3,dim = 1)
    distance1 = distance(pred1,pred2)
    distance2 = distance(pred2,pred3)
    distance3 = distance(pred3,pred1)
    distance_all = distance1 + distance2 + distance3
    return torch.sub(criterion(output,label), distance_all,alpha = alpha).sum()/256
optimizer = optim.AdamW(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 3,eta_min=2e-6)

#load_data
xtrain = np.load("X_train_bandpass.npy")
xval = np.load("X_val_bandpass.npy")
xtest = np.load("X_test_bandpass.npy")
xtrain = torch.from_numpy(xtrain)
xval = torch.from_numpy(xval)
xtest = torch.from_numpy(xtest)

dwt_train = np.load("dwt_train.npy")
dwt_val = np.load("dwt_val.npy")
dwt_test = np.load("dwt_test.npy")
dwt_train = torch.from_numpy(dwt_train)
dwt_val = torch.from_numpy(dwt_val)
dwt_test = torch.from_numpy(dwt_test)

stft_train = np.load("stft_train.npy")
stft_val = np.load("stft_val.npy")
stft_test = np.load("stft_test.npy")
stft_train = torch.from_numpy(stft_train)
stft_val = torch.from_numpy(stft_val)
stft_test = torch.from_numpy(stft_test)

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

train_data = DataLoader(TensorDataset(xtrain,dwt_train,stft_train,sletrain,ytrain), batch_size=batch_size, shuffle=True)
val_data = DataLoader(TensorDataset(xval,dwt_val,stft_val,sleval,yval), batch_size=batch_size, shuffle=False)
test_data = DataLoader(TensorDataset(xtest,dwt_test,stft_test,sletest,ytest), batch_size=batch_size, shuffle=False)

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
max_count = 1
num_allow_end_condition = 14
for t in range(1):
    for epoch in tqdm(range(num_eporch)):
        net.train()
        print("Epoch {}:/n-----------------------------------------------------------".format(epoch +1)) 
        if epoch%4 == 3 and epoch < num_allow_end_condition:
            scheduler.step()
        train_acc = []
        train_loss = []
        train_auc = []
        train_score = []
        for inputs,dwt,stft,metadata,labels in tqdm(train_data):
            inputs,dwt,stft,metadata,labels = inputs.to(device),dwt.to(device),stft.to(device),metadata.to(device).float(),labels.to(device)
            x = [inputs.float(),dwt.float(),stft.float()]
            running_loss = 0.0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x,metadata)
            if e2e:
                loss = criterion_func(outputs, labels.float(), net.features[0],net.features[1],net.features[2])
            else:
                loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            # calculate output accuracy and f1 score
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()
            score = f1_score(labels,pred,average='macro')
            area = roc_auc_score(labels,outputs.cpu().detach().numpy(),average = "macro")
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
            net.eval()
            running_loss = 0.0
            for inputs,dwt,stft,metadata,labels in tqdm(val_data):
                inputs,dwt,stft,metadata,labels = inputs.to(device),dwt.to(device),stft.to(device),metadata.to(device).float(),labels.to(device)
                x = [inputs.float(),dwt.float(),stft.float()]
                # forward + backward + optimize
                outputs = net(x,metadata)
                if e2e:
                    loss = criterion_func(outputs, labels.float(), net.features[0],net.features[1],net.features[2])
                else:
                    loss = criterion(outputs, labels.float())
                # calculate output acc
                pred = outputs.cpu().detach().numpy() > thresh_hold
                labels = labels.cpu().detach().numpy()
                score = f1_score(labels,pred,average='macro')
                area = roc_auc_score(labels,outputs.cpu().detach().numpy(),average = "macro")
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
            if epoch >= num_allow_end_condition:    
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
        net.eval()
        running_loss = 0.0
        batch_test = []
        for inputs,dwt,stft,metadata,labels in tqdm(test_data):
            inputs,dwt,stft,metadata,labels = inputs.to(device),dwt.to(device),stft.to(device),metadata.to(device).float(),labels.to(device)
            x = [inputs.float(),dwt.float(),stft.float()]
            outputs = net(x,metadata)
            if e2e:
                loss = criterion_func(outputs, labels.float(), net.features[0],net.features[1],net.features[2])
            else:
                loss = criterion(outputs, labels.float())
            # calculate output acc
            pred = outputs.cpu().detach().numpy() > thresh_hold
            labels = labels.cpu().detach().numpy()
            score = f1_score(labels,pred,average='macro')
            area = roc_auc_score(labels,outputs.cpu().detach().numpy(),average = "macro")
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

torch.save(net.classify.state_dict(),"ensemble_classify_test")