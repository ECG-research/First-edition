import numpy as np
import wfdb
import pandas as pd
import ast
import cv2
import skimage
# import tftb
import scipy as scp
from scipy import signal as sg
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pywt

path = "/home/ubuntu/Tue.CM210908/data/physionet.org/files/ptb-xl/1.0.3/"
sampling_rate = 100
cm_jet = cm.get_cmap('jet')

class Preprocessing():
    def __init__(self,path,sampling_rate,experiment = 'diagnostic_superclass'):
        self.experiment = experiment
        self.path = path
        self.sampling_rate = sampling_rate
        self.csv_file = pd.read_csv(path+'ptbxl_database.csv',index_col = 'ecg_id')
        self.csv_file.scp_codes = self.csv_file.scp_codes.apply(lambda x: ast.literal_eval(x))
        
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
        self.meta_sle[self.csv_file.age == 300, 31] = 1

        #uncomment the following block for SLE

        def check_not_empty(metadata):
            bool1 = 1 in metadata[2:11]
            bool2 = 1 in metadata[12:21]
            return [bool1, bool2]

        def fill_noise(metadata):
            bool = check_not_empty(metadata)
            male_height =   [0,7,7,7,7,7,7,7,7,7]
            female_height = [1,6,6,6,6,6,6,6,6,5]
            male_weight =   [1,6,6,6,6,6,6,6,5,5]
            female_weight = [1,5,5,5,5,5,5,5,4,4]
            if not bool[0]:
                if metadata[0] == 1:
                    l = female_height
                else: l = male_height
                for i in range(22,32):
                    if metadata[i] == 1:
                        break
                metadata[l[i-22]+2] = 1
            if not bool[1]:
                if metadata[0] == 1:
                    l2 = female_weight
                else: l2 = male_weight
                for j in range(22,32): #age indicator
                    if metadata[j] == 1:
                        break
                metadata[l2[j-22]+12] = 1
            return metadata        

        for i in range(21799):
            self.meta_sle[i] = fill_noise(self.meta_sle[i])

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
        #if using the baseline models, please reshape y_train,y_val and y_test into (-1,self.num_of_class)

        #có thể comment out đống này nếu đã có file X_train.npy
        X = self.load_raw_data(self.csv_file, self.sampling_rate, self.path)
        self.X_train = X[np.where(self.csv_file.strat_fold <= 8)]
        self.X_val = X[np.where(self.csv_file.strat_fold == 9)]
        self.X_test = X[np.where(self.csv_file.strat_fold == 10)]
        self.X_train = self.split_reshape(self.X_train)
        self.X_val = self.split_reshape(self.X_val)
        self.X_test = self.split_reshape(self.X_test)

        self.y_train = self.y[np.where(self.csv_file.strat_fold <= 8)]
        self.y_train = np.reshape(self.y_train,(-1,self.num_of_class))
        self.sle_train = self.meta_sle[np.where(self.csv_file.strat_fold <= 8)]
        self.sle_train = np.reshape(self.sle_train,(-1,32,1))

        self.y_train, index = self.remove_wrong_labels(self.y_train)
        self.X_train = self.remove_wrong_labels(self.X_train, index)
        self.sle_train = self.remove_wrong_labels(self.sle_train,index)

        #extract data       
        self.y_val = self.y[np.where(self.csv_file.strat_fold == 9)]
        self.y_val = np.reshape(self.y_val,(-1,self.num_of_class))
        self.sle_val = self.meta_sle[np.where(self.csv_file.strat_fold == 9)]
        self.sle_val = np.reshape(self.sle_val,(-1,32,1))

        self.y_val, index = self.remove_wrong_labels(self.y_val)
        self.X_val = self.remove_wrong_labels(self.X_val, index)
        self.sle_val = self.remove_wrong_labels(self.sle_val,index)

        self.y_test = self.y[np.where(self.csv_file.strat_fold == 10)]
        self.y_test = np.reshape(self.y_test,(-1,self.num_of_class))
        self.sle_test = self.meta_sle[np.where(self.csv_file.strat_fold == 10)]
        self.sle_test = np.reshape(self.sle_test,(-1,32,1))

        self.y_test, index = self.remove_wrong_labels(self.y_test)
        self.X_test = self.remove_wrong_labels(self.X_test, index)
        self.sle_test = self.remove_wrong_labels(self.sle_test,index)

    
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
        # join all leads
        inputs = np.concatenate(inputs, axis =1)
        return inputs

    def remove_wrong_labels(self,y,index = None):
        if index is None:    
            check = np.sum(y,axis = 1)
            index = check == 0
            return np.delete(y,index,axis = 0), index
        else:
            return np.delete(y,index,axis = 0)

    def get_data_x(self):
        return self.X_train,self.X_val,self.X_test
    
    def get_fast_data_x(self):
        return np.load("X_train.npy"),np.load("X_val.npy"),np.load("X_test.npy")
    
    def get_data_cwt(self):
        return np.load("cwt_train.npy"), np.load("cwt_val.npy"), np.load("cwt_test.npy")
    
    def get_data_stft(self):
        return np.load("stft_train.npy"), np.load("stft_val.npy"), np.load("stft_test.npy")
    
    def get_data_y(self):
        return self.y_train,self.y_val,self.y_test
    def get_data_metadata(self):
        return self.sle_train,self.sle_val,self.sle_test

def bandpass(data, lead_index, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = False):
    n, wn = scp.signal.buttord(bp, sp, gpass, spass, fs = fs)
    b, a = scp.signal.butter(n, wn, 'bandpass', fs = fs)
    y = scp.signal.lfilter(b, a, data[lead_index])
    if median:
        kernel = np.array([1,5,1])
        y = scp.ndimage.median_filter(y,footprint = kernel, mode = "nearest" )
    return np.asarray(y)
def convert_bandpass(data, median = False):
    for i in range(np.shape(data)[0]): 
        data[i][0] = bandpass(data[i], 0, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][1] = bandpass(data[i], 1, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][2] = bandpass(data[i], 2, bp = [1,41], sp = [0,48], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
    return data
    
def get_img_from_fig(fig, dpi=64):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi,bbox_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[35:275,50:370]
        return np.dot(img, [0.2989, 0.5870, 0.1140])

def stft_(x,nperseg=100,size = (224,224),rgb=False):
        
        f,t,Zxx = scp.signal.stft(x,fs=100,nperseg=nperseg)
        bruh = np.abs(Zxx)
        bruh = cv2.resize(bruh,size)
        bruh = (bruh-np.min(bruh))/(np.max(bruh)-np.min(bruh))
        
        if (rgb):
            bruh = cm_jet(bruh)
            bruh = np.uint8(bruh*255) #bỏ dòng này để chạy về khoảng [0,1]
            bruh = bruh[:,:,0:3].transpose(2,0,1) #lấy rgb và đổi về (3,224,224)
        return bruh

def cwt_gray(signal,f_min=1,f_max=47, size = (224,224),start=0,finish=500,rgb=False):
    scales = pywt.central_frequency('morl') * sampling_rate / np.arange(f_min, f_max + 1, 1)
    scales = np.arange(1,48)
    coef, freqs=pywt.cwt(signal[start:finish],scales,'morl',sampling_period=0.01)
    bruh = np.abs(coef)
    bruh = (bruh-np.min(bruh))/(np.max(bruh)-np.min(bruh))
    bruh = cv2.resize(bruh,size)
    if rgb:
        bruh = cm_jet(bruh)
        bruh = np.uint8(bruh*255) #có thể bỏ đi ở đây để cho nó về 01
        bruh = bruh[:,:,0:3].transpose(2,0,1) #lấy rgb và đổi về (3,224,224)
    return bruh 

def _cwt_scipy(sig,widths):
    return sg.cwt(sig, sg.ricker, widths)

def dwt_(x, wavelet = "db1", mode = "cpd"):
    (cA, cD) = pywt.dwt(x,wavelet,mode = mode)
    cA = np.squeeze(cA) 
    cD = np.squeeze(cD) 
    return np.concatenate((cA,cD), axis = 1)

def wvd_raw(x):
        wvd = tftb.processing.WignerVilleDistribution(x,timestamps=np.arange(1000)*0.01)
        tfr_wvd, t_wvd, f_wvd = wvd.run()
        #nếu có thể, t muốn pooling trước khi mình xử lí đống kia
        return tfr_wvd
    
def stft_data(xtrain, xval, xtest, nperseg = 100, rgb=False):
    shape = (224,224)
    rgb = True
    if not rgb:
        stft_train = np.zeros((xtrain.shape[0], xtrain.shape[1], shape[0], shape[1]))
        stft_val = np.zeros((xval.shape[0], xval.shape[1], shape[0], shape[1]))
        stft_test = np.zeros((xtest.shape[0], xtest.shape[1], shape[0], shape[1]))
        for i in range(xtrain.shape[0]):
            for j in range(3):
                stft_train[i][j] = stft_(xtrain[i][j],rgb=rgb,nperseg=nperseg)
        for i in range(xval.shape[0]):
            for j in range(3):
                stft_val[i][j] = stft_(xval[i][j],rgb=rgb,nperseg=nperseg)
        for i in range(xtest.shape[0]):
            for j in range(3):
                stft_test[i][j] = stft_(xtest[i][j],rgb=rgb,nperseg=nperseg)
    
    elif rgb:
        stft_train = np.zeros((xtrain.shape[0], 3*xtrain.shape[1], shape[0], shape[1]))
        stft_val = np.zeros((xval.shape[0], 3*xval.shape[1], shape[0], shape[1]))
        stft_test = np.zeros((xtest.shape[0], 3*xtest.shape[1], shape[0], shape[1]))
        for i in range(xtrain.shape[0]):
            for j in range(3):
                stft_train[i][3*j:3*j+3] = stft_(xtrain[i][j],rgb=rgb,nperseg=nperseg)
        for i in range(xval.shape[0]):
            for j in range(3):
                stft_val[i][3*j:3*j+3] = stft_(xval[i][j],rgb=rgb,nperseg=nperseg)
        for i in range(xtest.shape[0]):
            for j in range(3):
                stft_test[i][3*j:3*j+3] = stft_(xtest[i][j],rgb=rgb,nperseg=nperseg)
        #ha, hope this works
    np.save("stft_train",stft_train), np.save("stft_val",stft_val), np.save("stft_test",stft_test)
    print("stft done")

def cwt_data(xtrain, xval, xtest,f_min=1,f_max=47,size = (224,224),start=0,finish=500,rgb = False):
    rgb = False
    if not rgb:
        cwt_train = np.ndarray((xtrain.shape[0], xtrain.shape[1], size[0], size[1]))
        cwt_val =   np.ndarray((xval.shape[0],   xval.shape[1],   size[0], size[1]))
        cwt_test =  np.ndarray((xtest.shape[0],  xtest.shape[1],  size[0], size[1]))
        for i in range(xtrain.shape[0]):
            for j in range(3):
                cwt_train[i][j] = cwt_gray(xtrain[i][j],f_min,f_max,size,start,finish,rgb=rgb)
        for i in range(xval.shape[0]):
            for j in range(3):
                cwt_val[i][j] = cwt_gray(xval[i][j],f_min,f_max,size,start,finish,rgb=rgb)
        for i in range(xtest.shape[0]):
            for j in range(3):
                cwt_test[i][j] = cwt_gray(xtest[i][j],f_min,f_max,size,start,finish,rgb=rgb)
    elif rgb:
        cwt_train = np.ndarray((xtrain.shape[0], 3*xtrain.shape[1], size[0], size[1]))
        cwt_val =   np.ndarray((xval.shape[0],   3*xval.shape[1],   size[0], size[1]))
        cwt_test =  np.ndarray((xtest.shape[0],  3*xtest.shape[1],  size[0], size[1]))
        for i in range(xtrain.shape[0]):
            for j in range(3):
                cwt_train[i][3*j:3*j+3] = cwt_gray(xtrain[i][j],f_min,f_max,size,start,finish,rgb=rgb)
        for i in range(xval.shape[0]):
            for j in range(3):
                cwt_val[i][3*j:3*j+3] = cwt_gray(xval[i][j],f_min,f_max,size,start,finish,rgb=rgb)
        for i in range(xtest.shape[0]):
            for j in range(3):
                cwt_test[i][3*j:3*j+3] = cwt_gray(xtest[i][j],f_min,f_max,size,start,finish,rgb=rgb)        
    np.save("cwt_train",cwt_train), np.save("cwt_val",cwt_val), np.save("cwt_test",cwt_test)
    print(cwt_train.shape)
    print("cwt done")

def cwt_data_scipy(xtrain, xval, xtest, widths = np.arange(1,31), size = (30,250)):
    cwt_train = np.ndarray((xtrain.shape[0], xtrain.shape[1], size[0], size[1]))
    cwt_val =   np.ndarray((xval.shape[0],   xval.shape[1],   size[0], size[1]))
    cwt_test =  np.ndarray((xtest.shape[0],  xtest.shape[1],  size[0], size[1]))
    for i in range(xtrain.shape[0]):
        for j in range(3):
            cwt_train[i][j] = _cwt_scipy(xtrain[i][j],widths)
    for i in range(xval.shape[0]):
        for j in range(3):
            cwt_val[i][j] = _cwt_scipy(xval[i][j],widths)
    for i in range(xtest.shape[0]):
        for j in range(3):
            cwt_test[i][j] = _cwt_scipy(xtest[i][j],widths)
    np.save("cwt_train_ricker",cwt_train), np.save("cwt_val_ricker",cwt_val), np.save("cwt_test_ricker",cwt_test)
    print("cwt done")

def wvd_data(xtrain, xval, xtest):
    wvd_train = np.zeros((xtrain.shape[0], xtrain.shape[1], 1000, 1000))
    wvd_val =   np.zeros((xval.shape[0], xval.shape[1], 1000, 1000))
    wvd_test =  np.zeros((xtest.shape[0], xtest.shape[1], 1000, 1000))
    for i in range(xtrain.shape[0]):
        for j in range(3):
            wvd_train[i][j] = wvd_raw(xtrain[i][j])
    for i in range(xval.shape[0]):
        for j in range(3):
            wvd_val[i][j] = wvd_raw(xval[i][j])
    for i in range(xtest.shape[0]):
        for j in range(3):
            wvd_test[i][j] = wvd_raw(xtest[i][j])
    np.save("wvd_train",wvd_train), np.save("wvd_val",wvd_val), np.save("wvd_test",wvd_test)
    print("wvd done")

def dwt_data(xtrain, xval, xtest, wavelet = "db1", mode = "cpd"):
    coeffs_train = []
    coeffs_val = []
    coeffs_test = []
    for i in range(xtrain.shape[0]):
        coeff = dwt_(xtrain[i], wavelet, mode)
        coeffs_train.append(coeff)
    for i in range(xval.shape[0]):
        coeff = dwt_(xval[i], wavelet, mode)
        coeffs_val.append(coeff)
    for i in range(xtest.shape[0]):
        coeff = dwt_(xtest[i], wavelet, mode)
        coeffs_test.append(coeff)
    np.save("dwt_train",coeffs_train), np.save("dwt_val",coeffs_val), np.save("dwt_test",coeffs_test)
    print("dwt done")

def segmentation(x,y,sle,n_part = 4):
    assert 1000%n_part == 0, "n_part must be the divisor of squence length"
    x_aug = np.split(x, n_part, axis = 2)
    y_aug = []
    sle_aug = []
    for _ in range(n_part):    
        y_aug.append(y)
        sle_aug.append(sle)
    return np.concatenate(x_aug,axis = 0), np.concatenate(y_aug,axis = 0), np.concatenate(sle_aug,axis = 0)


# P = Preprocessing(sampling_rate=100,path=path)
# X_train,X_val,X_test = P.get_data_x()
# xtrain, xval, xtest = X_train, X_val, X_test

# xtrain = np.load("X_train_bandpass.npy")
# xval = np.load("X_val_bandpass.npy")
# xtest = np.load("X_test_bandpass.npy")

ytrain = np.load("ytrain.npy")
yval = np.load("yval.npy")
ytest = np.load("ytest.npy")

# sletrain = np.load("sletrain.npy")
# sleval = np.load("sleval.npy")
# sletest = np.load("sletest.npy")

print(ytrain.shape)

# np.save("X_train",xtrain)
# np.save("X_val",xval)
# np.save("X_test",xtest)

# ytrain,yval,ytest = P.get_data_y()
# sletrain,sleval,sletest = P.get_data_metadata()

# xtrain_bandpass = convert_bandpass(xtrain)
# xtest_bandpass = convert_bandpass(xtest)
# xval_bandpass = convert_bandpass(xval)

# xtrain_bandpass, ytrain, sletrain = segmentation(xtrain_bandpass, ytrain, sletrain)
# xval_bandpass, yval, sleval = segmentation(xval_bandpass, yval, sleval)
# xtest_bandpass, ytest, sletest = segmentation(xtest_bandpass, ytest, sletest)

# np.save("X_train_bandpass",xtrain_bandpass)
# np.save("X_val_bandpass",xval_bandpass)
# np.save("X_test_bandpass",xtest_bandpass)


# ytrain,yval,ytest = P.get_data_y()
# np.save("ytrain",ytrain), np.save("yval",yval), np.save("ytest",ytest)

# sletrain,sleval,sletest = P.get_data_metadata()
# np.save("sletrain",sletrain), np.save("sleval",sleval), np.save("sletest",sletest)

# cwt_data_scipy(xtrain,xval,xtest)
# stft_data(xtrain,xval,xtest)
# dwt_data(xtrain,xval,xtest)