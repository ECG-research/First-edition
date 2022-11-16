import numpy as np
import wfdb
import pandas as pd
import ast
import Model as M
import tensorflow as tf
path = "First-edition\\"
sampling_rate = 100

class Preprocessing():
    def __init__(self,path,sampling_rate,experiment = 'diagnostic_superclass'):
        self.experiment = experiment
        self.path = path
        self.sampling_rate = sampling_rate
        self.csv_file = pd.read_csv(path+'ptbxl_database.csv',index_col = 'ecg_id')
        self.csv_file.scp_codes = self.csv_file.scp_codes.apply(lambda x: ast.literal_eval(x))
        X = self.load_raw_data(self.csv_file, self.sampling_rate, self.path)
        self.num_of_class = 5
        if (self.experiment == 'diagnostic_subclass'):
            self.num_of_class = 23

        self.agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
        # Apply diagnostic superclass
        self.csv_file['diagnostic_superclass'] = self.csv_file.scp_codes.apply(self.aggregate_diagnostic)
        self.csv_file['diagnostic_subclass'] = self.csv_file.scp_codes.apply(self.aggregate_subdiagnostic)
        # Split data into train and test
        self.test_fold = 10
        self.val_fold = 9

        #soft_label_encoding
        self.meta_sle = np.full((21799,32),0.1)
        self.meta_sle[self.csv_file.sex==1,0] = 1
        self.meta_sle[self.csv_file.sex==0,1] = 1
        for i in range(10):
            self.meta_sle[np.logical_and(np.array(self.csv_file.height)>=(100+10*i), np.array(self.csv_file.height)<(110+10*i)), 2+i] = 1 #height encoding
            self.meta_sle[np.logical_and(np.array(self.csv_file.weight)>=(0+12*i), np.array(self.csv_file.weight)<(12+12*i)), 12+i] = 1 #weight encoding
            self.meta_sle[np.logical_and(np.array(self.csv_file.age)>=(0+10*i), np.array(self.csv_file.age)<(10+10*i)), 22+i] = 1 #age encoding

        #one hot coding for y
        self.y = np.zeros((21799,5))
        if (self.experiment == "diagnostic_superclass"):
            l1 = ['NORM' in i for i in self.csv_file.diagnostic_superclass]
            l2 = ['MI' in i for i in self.csv_file.diagnostic_superclass]
            l3 = ['HYP' in i for i in self.csv_file.diagnostic_superclass]
            l4 = ['STTC' in i for i in self.csv_file.diagnostic_superclass]
            l5 = ['CD' in i for i in self.csv_file.diagnostic_superclass]
            self.y[l1,0] = 1
            self.y[l2,1] = 1
            self.y[l3,2] = 1
            self.y[l4,3] = 1
            self.y[l5,4] = 1
        elif (self.experiment == "diagnostic_subclass"):
            self.y = np.zeros((21799,23))
            self.l = ['_AVB','AMI','CLBBB','CRBBB','ILBBB','IMI','IRBBB','ISC_','ISCA','ISCI','IVCD','LAFB/LPFB','LAO/LAE','LMI','LVH','NORM','NST_','PMI','RAO/RAE','RVH','SEHYP','STTC','WPW']
            for j in range(23):
                l_j = [self.l[j] in i for i in self.csv_file.diagnostic_subclass]
                self.y[l_j, j] = 1

        #meta_sle 
        self.X_train = X[np.where(self.csv_file.strat_fold <= 8)]
        self.y_train = self.y[np.where(self.csv_file.strat_fold <= 8)]
        self.y_train = np.reshape(self.y_train,(-1,1,self.num_of_class))
        self.sle_train = self.meta_sle[np.where(self.csv_file.strat_fold <= 8)]
        self.sle_train = np.reshape(self.sle_train,(-1,1,32))

        #extract data
        self.X_val = X[np.where(self.csv_file.strat_fold == 9)]
        self.y_val = self.y[np.where(self.csv_file.strat_fold == 9)]
        self.y_val = np.reshape(self.y_val,(-1,1,self.num_of_class))
        self.sle_val = self.meta_sle[np.where(self.csv_file.strat_fold == 9)]
        self.sle_val = np.reshape(self.sle_val,(-1,1,32))

        self.X_test = X[np.where(self.csv_file.strat_fold == 10)]
        self.y_test = self.y[np.where(self.csv_file.strat_fold == 10)]
        self.y_test = np.reshape(self.y_test,(-1,1,self.num_of_class))
        self.sle_test = self.meta_sle[np.where(self.csv_file.strat_fold == 10)]
        self.sle_test = np.reshape(self.sle_test,(-1,1,32))

        self.X_train = self.split_reshape(self.X_train)
        self.X_val = self.split_reshape(self.X_val)
        self.X_test = self.split_reshape(self.X_test)

        self.X_train = self.add_position_temporal(self.X_train)
        self.X_val = self.add_position_temporal(self.X_val)
        self.X_test = self.add_position_temporal(self.X_test)
    
    def load_raw_data(self,df,sampling_rate,path):
        if (sampling_rate == 100):
            data = [wfdb.rdsamp(path+ f,channels=[0,1,7]) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f,channels=[0,1,7]) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_diagnostic(self,y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_subdiagnostic(self,y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_subclass)
        return list(set(tmp))


    #reshape dataset as model dimention for input
    def split_reshape(self,input):
        inputs = []
        input1, input2, input3 = np.split(input,input.shape[2],axis=2)
        input1 = np.reshape(input1,(-1,1,input.shape[1]))
        input2 = np.reshape(input2,(-1,1,input.shape[1]))
        input3 = np.reshape(input3,(-1,1,input.shape[1]))
        inputs.append(input1)
        inputs.append(input2)
        inputs.append(input3)
        return inputs

    #add temporal position
    def add_position_temporal(self,input):
        for j in range(len(input)):
            z = np.full((1,1,input[j].shape[2]),range(input[j].shape[2]))       
            for i in range(input[j].shape[0]): 
                input[j][i,:,:] = np.add(input[j][i,:,:] ,z)
            return input

    def get_data_x(self):
        return self.X_train,self.X_val,self.X_test
    def get_data_y(self):
        return self.y_train,self.y_val,self.y_test
    def get_data_metadata(self):
        return self.sle_train,self.sle_val,self.sle_test
    
