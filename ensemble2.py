import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import second_edition as model
# torch.use_deterministic_algorithms(True)
#hyperparameters
path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
num_eporch = 100
thresh_hold = 0.5
batch_size = 128
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# transfering pseudoclassifier
p_classifier = model.Pseudo_classifier().to(device)
p_classifier.load_state_dict(torch.load("pseudo_test"))
p_classifier.eval()
for param in p_classifier.parameters():
    param.requires_grad = False

def load_pm(path, type):
    pm = model.Projection_Module(type = type)
    pm.load_state_dict(torch.load(path))
    pm.eval()
    for param in pm.parameters():
        param.requires_grad = False
    return pm

#load projection module
pm_raw = load_pm("raw_pm_test","raw").to(device)
pm_dwt = load_pm("dwt_pm_test","raw").to(device) # type is raw because I used inception for dwt
pm_stft = load_pm("stft_pm_test","stft").to(device)
pms = [pm_raw,pm_dwt,pm_stft]

sd = model.Second_edition(pms)
ensemble = sd.classify
ensemble.load_state_dict(torch.load("ensemble_classify_test"))
ensemble.eval().to(device)
for param in ensemble.parameters():
        param.requires_grad = False

#initialize criterion, optimizer
criterion = nn.functional.binary_cross_entropy_with_logits

#load_data
# xtrain = np.load("X_train_bandpass.npy")
xval = np.load("X_val_bandpass.npy")
xtest = np.load("X_test_bandpass.npy")
# xtrain = torch.from_numpy(xtrain)
xval = torch.from_numpy(xval)
xtest = torch.from_numpy(xtest)

# dwt_train = np.load("dwt_train.npy")
dwt_val = np.load("dwt_val.npy")
dwt_test = np.load("dwt_test.npy")
# dwt_train = torch.from_numpy(dwt_train)
dwt_val = torch.from_numpy(dwt_val)
dwt_test = torch.from_numpy(dwt_test)

# stft_train = np.load("stft_train.npy")
stft_val = np.load("stft_val.npy")
stft_test = np.load("stft_test.npy")
# stft_train = torch.from_numpy(stft_train)
stft_val = torch.from_numpy(stft_val)
stft_test = torch.from_numpy(stft_test)

# ytrain = np.load("ytrain.npy")
yval = np.load("yval.npy")
ytest = np.load("ytest.npy")
# ytrain = torch.from_numpy(ytrain)
yval = torch.from_numpy(yval)
ytest = torch.from_numpy(ytest)

# sletrain = np.load("sletrain.npy")
sleval = np.load("sleval.npy")
sletest = np.load("sletest.npy")
# sletrain = torch.from_numpy(sletrain)
sleval = torch.from_numpy(sleval)
sletest = torch.from_numpy(sletest)

# train_data = DataLoader(TensorDataset(xtrain,dwt_train,stft_train,sletrain,ytrain), batch_size=batch_size, shuffle=True)
val_data = DataLoader(TensorDataset(xval,dwt_val,stft_val,sleval,yval), batch_size=batch_size, shuffle=False)
test_data = DataLoader(TensorDataset(xtest,dwt_test,stft_test,sletest,ytest), batch_size=batch_size, shuffle=False)

#train
def get_pred(input,metadata,type):
    if type == "raw":
        out = pm_raw(input.float(),metadata.float())
        pred = p_classifier(out)
    elif type == "dwt":
        out = pm_dwt(input.float(),metadata.float())
        pred = p_classifier(out)
    elif type == "stft":
        out = pm_stft(input.float(),metadata.float())
        pred = p_classifier(out)
    else:
        raise "wrong type"
    return out, pred

def merge(outs,kernel):
    outputs = torch.zeros_like(outs[0])
    for i in range(len(outs)):
        outputs += outs[i]*kernel[i]/sum(kernel)
    return outputs

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
kernel = [6,5,4,2]
with torch.no_grad():
        val_loss = []
        val_acc = []
        val_auc = []
        val_score = []
        running_loss = 0.0
        for inputs,dwt,stft,metadata,labels in tqdm(val_data):
            inputs,dwt,stft,metadata,labels = inputs.to(device),dwt.to(device),stft.to(device),metadata.to(device).float(),labels.to(device)
            x = [inputs.float(),dwt.float(),stft.float()]
            # forward + backward + optimize
            out1, pred1 = get_pred(inputs,metadata,"raw")
            out2, pred2 = get_pred(dwt,metadata,"dwt")
            out3, pred3 = get_pred(stft,metadata,"stft")
            ens_input = torch.concatenate((out1,out2,out3),dim = 1)
            out_ens = ensemble(ens_input)
            outputs = merge([pred1,out_ens,pred2,pred3],kernel)

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
        val_loss_all.append(epoch_val_loss)
        val_acc_all.append(epoch_val_acc)
        val_auc_all.append(epoch_val_auc)
        val_score_all.append(epoch_val_score)
        print("========================================================")
        print("Validation loss of epoch :{:.4f}".format(epoch_val_loss))
        print("Validation accuracy of epoch :{}".format(epoch_val_acc))
        print("Validation F1 score of epoch :{}".format(epoch_val_score))
        print("Validation AUC score of epoch :{}".format(epoch_val_auc))
test_loss = []
test_score = []
test_acc = []
test_auc = []
with torch.no_grad():
    running_loss = 0.0
    batch_test = []
    for inputs,dwt,stft,metadata,labels in tqdm(test_data):
        inputs,dwt,stft,metadata,labels = inputs.to(device),dwt.to(device),stft.to(device),metadata.to(device).float(),labels.to(device)
        x = [inputs.float(),dwt.float(),stft.float()]
        # forward
        out1, pred1 = get_pred(inputs,metadata,"raw")
        out2, pred2 = get_pred(dwt,metadata,"dwt")
        out3, pred3 = get_pred(stft,metadata,"stft")
        ens_input = torch.concatenate((out1,out2,out3),dim = 1)
        out_ens = ensemble(ens_input)
        outputs = merge([pred1,out_ens,pred2,pred3],kernel)
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