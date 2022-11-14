import numpy as np
import wfdb
import pandas as pd
import ast
import Model as M
import tensorflow as tf
path = "First-edition\\"
sampling_rate = 100

def load_raw_data(df,sampling_rate,path):
    if (sampling_rate == 100):
        data = [wfdb.rdsamp(path+ f,channels=[0,1,7]) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f,channels=[0,1,7]) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

csv_file = pd.read_csv(path+'ptbxl_database.csv',index_col = 'ecg_id')
csv_file.scp_codes = csv_file.scp_codes.apply(lambda x: ast.literal_eval(x))
X = load_raw_data(csv_file, sampling_rate, path)

agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
csv_file['diagnostic_superclass'] = csv_file.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
val_fold = 9

#soft_label_encoding
meta_sle = np.full((21799,32),0.1)
meta_sle[csv_file.sex==1,0] = 1
meta_sle[csv_file.sex==0,1] = 1

for i in range(10):
    meta_sle[np.logical_and(np.array(csv_file.height)>=(100+10*i), np.array(csv_file.height)<(110+10*i)), 2+i] = 1 #height encoding
    meta_sle[np.logical_and(np.array(csv_file.weight)>=(0+12*i), np.array(csv_file.weight)<(12+12*i)), 12+i] = 1 #weight encoding
    meta_sle[np.logical_and(np.array(csv_file.age)>=(0+10*i), np.array(csv_file.age)<(10+10*i)), 22+i] = 1 #age encoding

#one hot coding for y
y = np.zeros((21799,5))
l1 = ['NORM' in i for i in csv_file.diagnostic_superclass]
l2 = ['MI' in i for i in csv_file.diagnostic_superclass]
l3 = ['HYP' in i for i in csv_file.diagnostic_superclass]
l4 = ['STTC' in i for i in csv_file.diagnostic_superclass]
l5 = ['CD' in i for i in csv_file.diagnostic_superclass]
y[l1,0] = 1
y[l2,1] = 1
y[l3,2] = 1
y[l4,3] = 1
y[l5,4] = 1

#meta_sle 
X_train = X[np.where(csv_file.strat_fold <= 8)]
y_train = y[np.where(csv_file.strat_fold <= 8)]
y_train = np.reshape(y_train,(-1,1,5))
sle_train = meta_sle[np.where(csv_file.strat_fold <= 8)]
sle_train = np.reshape(sle_train,(-1,1,32))

X_val = X[np.where(csv_file.strat_fold == 9)]
y_val = y[np.where(csv_file.strat_fold == 9)]
y_val = np.reshape(y_val,(-1,1,5))
sle_val = meta_sle[np.where(csv_file.strat_fold == 9)]
sle_val = np.reshape(sle_val,(-1,1,32))

X_test = X[np.where(csv_file.strat_fold == 10)]
y_test = y[np.where(csv_file.strat_fold == 10)]
y_test = np.reshape(y_test,(-1,1,5))
sle_test = meta_sle[np.where(csv_file.strat_fold == 10)]
sle_test = np.reshape(sle_test,(-1,1,32))

#reshape dataset as model dimention for input
def split_reshape(input):
    inputs = []
    input1, input2, input3 = np.split(input,input.shape[2],axis=2)
    input1 = np.reshape(input1,(-1,1,input.shape[1]))
    input2 = np.reshape(input2,(-1,1,input.shape[1]))
    input3 = np.reshape(input3,(-1,1,input.shape[1]))
    inputs.append(input1)
    inputs.append(input2)
    inputs.append(input3)
    return inputs
X_train = split_reshape(X_train)
X_val = split_reshape(X_val)
X_test = split_reshape(X_test)

#add temporal position
def add_position_temporal(input):
    for j in range(len(input)):
        z = np.full((1,1,input[j].shape[2]),range(input[j].shape[2]))       
        for i in range(input[j].shape[0]): 
            input[j][i,:,:] = np.add(input[j][i,:,:] ,z)
        return input
X_train = add_position_temporal(X_train)
X_val = add_position_temporal(X_val)
X_test = add_position_temporal(X_test)

model = M.Proposed_model(1000,3,5)
model.fit([X_train,sle_train],y_train,batch_size=32,validation_data = ([X_val,sle_val],y_val),epochs=1)
