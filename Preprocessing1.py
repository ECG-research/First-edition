import numpy as np
import wfdb
import pandas as pd
import ast
import cv2
# import skimage
# import cohen as ch
# import tftb
import scipy as scp
from scipy import signal as sg
import io
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import pywt
from sklearn import preprocessing
sampling_rate = 100
def init_everything():
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f,channels=[0,1,2,3,4,5,6,7,8,9,10,11]) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f,channels=[0,1,2,3,4,5,6,7,8,9,10,11]) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    path = ''
    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    def aggregate_subdiagnostic(self,y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_subclass)
        return list(set(tmp))
    def aggregate_all_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(key)
        return list(set(tmp))
    val_fold = 9
    test_fold = 10

    # Apply labels
    experiment = 'diag'
    if experiment == 'diagnostic_superclass':
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
        y_train = Y[Y.strat_fold <= 8].diagnostic_superclass
        y_val = Y[Y.strat_fold == 9].diagnostic_superclass
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    elif experiment == 'diagnostic_subclass':
        Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_subdiagnostic)
        y_train = Y[Y.strat_fold <= 8].diagnostic_subclass
        y_val = Y[Y.strat_fold == 9].diagnostic_subclass
        y_test = Y[Y.strat_fold == test_fold].diagnostic_subclass
    elif experiment == 'all':
        Y['all_scp'] = Y.scp_codes.apply(lambda x: list(set(x.keys())))
        counts = pd.Series(np.concatenate(Y.all_scp.values)).value_counts()
        counts = counts[counts >= 0]
        Y.all_scp = Y.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        y_train = Y[Y.strat_fold <= 8].all_scp
        y_val = Y[Y.strat_fold == 9].all_scp
        y_test = Y[Y.strat_fold == test_fold].all_scp
    elif experiment == 'diag':
        Y['diagnostic'] = Y.scp_codes.apply(aggregate_all_diagnostic)
        y_train = Y[Y.strat_fold <= 8].diagnostic
        y_val = Y[Y.strat_fold == 9].diagnostic
        y_test = Y[Y.strat_fold == test_fold].diagnostic
        
    X_train = X[np.where(Y.strat_fold <= 8)].transpose(0, 2, 1) 
    X_val = X[np.where(Y.strat_fold == 9)].transpose(0, 2, 1) 
    X_test = X[np.where(Y.strat_fold == test_fold)].transpose(0, 2, 1) 

    #label binarizing
    mlb = preprocessing.MultiLabelBinarizer()
    if experiment == "diagnostic_superclass":
        mlb.fit([['CD','HYP','MI','NORM','STTC']])
    elif experiment == "diagnostic_subclass":
        mlb.fit([['_AVB','AMI','CLBBB','CRBBB','ILBBB','IMI','IRBBB','ISC_','ISCA','ISCI','IVCD','LAFB/LPFB','LAO/LAE','LMI','LVH','NORM','NST_','PMI','RAO/RAE','RVH','SEHYP','STTC','WPW']])
    elif experiment == 'all':
        mlb.fit(Y.all_scp.values)
    elif experiment == 'diag':
        mlb.fit(Y.diagnostic.values)
    y_train = mlb.transform(y_train)
    y_val = mlb.transform(y_val)
    y_test = mlb.transform(y_test)
    #remove zeros labels
    remove_zeros = True
    if remove_zeros:
        a = []
        for y in y_train:
            if sum(y) == 0:
                a.append(False)
            else:
                a.append(True)
        X_train = X_train[a]
        y_train = y_train[a]
   
        a = []
        for y in y_val:
            if sum(y) == 0:
                a.append(False)
            else:
                a.append(True)
        X_val = X_val[a]
        y_val = y_val[a]

        a = []
        for y in y_test:
            if sum(y) == 0:
                a.append(False)
            else:
                a.append(True)
        X_test = X_test[a]
        y_test = y_test[a]

    # Apply bandpass filter

    # saving the data
    np.save("X_train", X_train)
    np.save("X_val",   X_val)
    np.save("X_test",  X_test)
    if experiment == "diagnostic_superclass":
        np.save("y_super_train", y_train)
        np.save("y_super_val",   y_val)
        np.save("y_super_test",  y_test)
    elif experiment == "diagnostic_subclass":
        np.save("y_sub_train", y_train)
        np.save("y_sub_val",   y_val)
        np.save("y_sub_test",  y_test)
    elif experiment == 'all':
        np.save("y_all_train", y_train)
        np.save("y_all_val",   y_val)
        np.save("y_all_test",  y_test)
    elif experiment == 'diag':
        np.save("y_diag_train", y_train)
        np.save("y_diag_val",   y_val)
        np.save("y_diag_test",  y_test)
    
    return X_train, X_val, X_test

xtrain,xval,xtest = init_everything()
print(xtrain.shape,xval.shape,xtest.shape)
#Transformations
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
        data[i][2] = bandpass(data[i], 2, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][3] = bandpass(data[i], 3, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][4] = bandpass(data[i], 4, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][5] = bandpass(data[i], 5, bp = [3,45], sp = [1,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][6] = bandpass(data[i], 6, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][7] = bandpass(data[i], 7, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][8] = bandpass(data[i], 8, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][9] = bandpass(data[i], 9, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][10] = bandpass(data[i], 10, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
        data[i][11] = bandpass(data[i], 11, bp = [1,40], sp = [0,50], gpass = 0.4, spass = 3, fs = sampling_rate, median = median)
    return data

# def get_img_from_fig(fig, dpi=64):
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", dpi=dpi,bbox_inches=0)
#         buf.seek(0)
#         img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#         buf.close()
#         img = cv2.imdecode(img_arr, 1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img[35:275,50:370]
#         return np.dot(img, [0.2989, 0.5870, 0.1140])

# def stft_(x,nperseg=40,size = (224,224),rgb=False):
        
#         f,t,Zxx = scp.signal.stft(x,fs=100,nperseg=nperseg)
#         bruh = np.abs(Zxx)
#         # bruh = cv2.resize(bruh,size)
#         bruh = (bruh-np.min(bruh))/(np.max(bruh)-np.min(bruh))
        
#         if (rgb):
#             bruh = cm_jet(bruh)
#             bruh = np.uint8(bruh*255) #bỏ dòng này để chạy về khoảng [0,1]
#             bruh = bruh[:,:,0:3].transpose(2,0,1) #lấy rgb và đổi về (3,224,224)
#         return bruh

# def cwt_gray(signal,f_min=1,f_max=47, size = (224,224),start=0,finish=500,rgb=False):
#     scales = pywt.central_frequency('morl') * sampling_rate / np.arange(f_min, f_max + 1, 1)
#     scales = np.arange(1,48)
#     coef, freqs=pywt.cwt(signal[start:finish],scales,'morl',sampling_period=0.01)
#     bruh = np.abs(coef)
#     bruh = (bruh-np.min(bruh))/(np.max(bruh)-np.min(bruh))
#     bruh = cv2.resize(bruh,size)
#     if rgb:
#         bruh = cm_jet(bruh)
#         bruh = np.uint8(bruh*255) #có thể bỏ đi ở đây để cho nó về 01
#         bruh = bruh[:,:,0:3].transpose(2,0,1) #lấy rgb và đổi về (3,224,224)
#     return bruh 

def _cwt_scipy(sig,widths):
    return sg.cwt(sig, sg.ricker, widths)

def dwt_(x, wavelet = "db1", mode = "cpd"):
    (cA, cD) = pywt.dwt(x,wavelet,mode = mode)
    cA = np.squeeze(cA) 
    cD = np.squeeze(cD) 
    return np.concatenate((cA,cD), axis = 1)

# def wvd_raw(x):
#         wvd = ch.WignerVilleDistribution(x,timestamps=np.arange(250)*0.01)
#         tfr_wvd, t_wvd, f_wvd = wvd.run()
#         #nếu có thể, t muốn pooling trước khi mình xử lí đống kia
#         return tfr_wvd

def fft_(x, axes = (2)):
    return scp.fft.fftn(x, axes=axes)

# def stft_data(xtrain, xval, xtest, nperseg = 40, rgb=False):
#     shape = (21,14)
#     if not rgb:
#         stft_train = np.zeros((xtrain.shape[0], xtrain.shape[1], shape[0], shape[1]))
#         stft_val = np.zeros((xval.shape[0], xval.shape[1], shape[0], shape[1]))
#         stft_test = np.zeros((xtest.shape[0], xtest.shape[1], shape[0], shape[1]))
#         for i in range(xtrain.shape[0]):
#             for j in range(3):
#                 stft_train[i][j] = stft_(xtrain[i][j],rgb=rgb,nperseg=nperseg)
#         for i in range(xval.shape[0]):
#             for j in range(3):
#                 stft_val[i][j] = stft_(xval[i][j],rgb=rgb,nperseg=nperseg)
#         for i in range(xtest.shape[0]):
#             for j in range(3):
#                 stft_test[i][j] = stft_(xtest[i][j],rgb=rgb,nperseg=nperseg)
    
#     elif rgb:
#         stft_train = np.zeros((xtrain.shape[0], 3*xtrain.shape[1], shape[0], shape[1]))
#         stft_val = np.zeros((xval.shape[0], 3*xval.shape[1], shape[0], shape[1]))
#         stft_test = np.zeros((xtest.shape[0], 3*xtest.shape[1], shape[0], shape[1]))
#         for i in range(xtrain.shape[0]):
#             for j in range(3):
#                 stft_train[i][3*j:3*j+3] = stft_(xtrain[i][j],rgb=rgb,nperseg=nperseg)
#         for i in range(xval.shape[0]):
#             for j in range(3):
#                 stft_val[i][3*j:3*j+3] = stft_(xval[i][j],rgb=rgb,nperseg=nperseg)
#         for i in range(xtest.shape[0]):
#             for j in range(3):
#                 stft_test[i][3*j:3*j+3] = stft_(xtest[i][j],rgb=rgb,nperseg=nperseg)
#         #ha, hope this works
#     np.save("stft_train",stft_train), np.save("stft_val",stft_val), np.save("stft_test",stft_test)
#     print("stft done")

# def cwt_data(xtrain, xval, xtest,f_min=1,f_max=47,size = (224,224),start=0,finish=500,rgb = False):
#     rgb = False
#     if not rgb:
#         cwt_train = np.ndarray((xtrain.shape[0], xtrain.shape[1], size[0], size[1]))
#         cwt_val =   np.ndarray((xval.shape[0],   xval.shape[1],   size[0], size[1]))
#         cwt_test =  np.ndarray((xtest.shape[0],  xtest.shape[1],  size[0], size[1]))
#         for i in range(xtrain.shape[0]):
#             for j in range(3):
#                 cwt_train[i][j] = cwt_gray(xtrain[i][j],f_min,f_max,size,start,finish,rgb=rgb)
#         for i in range(xval.shape[0]):
#             for j in range(3):
#                 cwt_val[i][j] = cwt_gray(xval[i][j],f_min,f_max,size,start,finish,rgb=rgb)
#         for i in range(xtest.shape[0]):
#             for j in range(3):
#                 cwt_test[i][j] = cwt_gray(xtest[i][j],f_min,f_max,size,start,finish,rgb=rgb)
#     elif rgb:
#         cwt_train = np.ndarray((xtrain.shape[0], 3*xtrain.shape[1], size[0], size[1]))
#         cwt_val =   np.ndarray((xval.shape[0],   3*xval.shape[1],   size[0], size[1]))
#         cwt_test =  np.ndarray((xtest.shape[0],  3*xtest.shape[1],  size[0], size[1]))
#         for i in range(xtrain.shape[0]):
#             for j in range(3):
#                 cwt_train[i][3*j:3*j+3] = cwt_gray(xtrain[i][j],f_min,f_max,size,start,finish,rgb=rgb)
#         for i in range(xval.shape[0]):
#             for j in range(3):
#                 cwt_val[i][3*j:3*j+3] = cwt_gray(xval[i][j],f_min,f_max,size,start,finish,rgb=rgb)
#         for i in range(xtest.shape[0]):
#             for j in range(3):
#                 cwt_test[i][3*j:3*j+3] = cwt_gray(xtest[i][j],f_min,f_max,size,start,finish,rgb=rgb)        
#     np.save("cwt_train",cwt_train), np.save("cwt_val",cwt_val), np.save("cwt_test",cwt_test)
#     print(cwt_train.shape)
#     print("cwt done")

def cwt_data_scipy(xtrain, xval, xtest, widths = np.arange(1,31), size = (30,1000)):
    cwt_train = np.ndarray((xtrain.shape[0], xtrain.shape[1], size[0], size[1]))
    cwt_val =   np.ndarray((xval.shape[0],   xval.shape[1],   size[0], size[1]))
    cwt_test =  np.ndarray((xtest.shape[0],  xtest.shape[1],  size[0], size[1]))
    for i in range(xtrain.shape[0]):
        for j in range(12):
            cwt_train[i][j] = _cwt_scipy(xtrain[i][j],widths)
    for i in range(xval.shape[0]):
        for j in range(12):
            cwt_val[i][j] = _cwt_scipy(xval[i][j],widths)
    for i in range(xtest.shape[0]):
        for j in range(12):
            cwt_test[i][j] = _cwt_scipy(xtest[i][j],widths)
    np.save("cwt_train_ricker_t",cwt_train), np.save("cwt_val_ricker_t",cwt_val), np.save("cwt_test_ricker_t",cwt_test)
    print("cwt done")

# def wvd_data(xtrain, xval, xtest):
#     wvd_train = np.zeros((xtrain.shape[0], xtrain.shape[1], 250, 250))
#     wvd_val =   np.zeros((xval.shape[0], xval.shape[1], 250, 250))
#     wvd_test =  np.zeros((xtest.shape[0], xtest.shape[1], 250, 250))
#     for i in range(xtrain.shape[0]):
#         for j in range(12):
#             wvd_train[i][j] = wvd_raw(xtrain[i][j])
#     for i in range(xval.shape[0]):
#         for j in range(12):
#             wvd_val[i][j] = wvd_raw(xval[i][j])
#     for i in range(xtest.shape[0]):
#         for j in range(12):
#             wvd_test[i][j] = wvd_raw(xtest[i][j])
#     np.save("wvd_train",wvd_train), np.save("wvd_val",wvd_val), np.save("wvd_test",wvd_test)
#     print("wvd done")

def dwt_data(xtrain, xval, xtest, wavelet = "haar", mode = "cpd"):
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

def fft_data(xtrain, xval, xtest):
    np.save("fft_train",fft_(xtrain)), np.save("fft_val",fft_(xval)), np.save("fft_test",fft_(xtest))
    print("fft done")

def segmentation(x,y,sle,n_part = 4):
    assert 1000%n_part == 0, "n_part must be the divisor of squence length"
    x_aug = np.split(x, n_part, axis = 2)
    x_aug.pop(-1), x_aug.pop(0)
    y_aug = []
    sle_aug = []
    for _ in range(n_part-2):    
        y_aug.append(y)
        sle_aug.append(sle)
    return np.concatenate(x_aug,axis = 0), np.concatenate(y_aug,axis = 0), np.concatenate(sle_aug,axis = 0)


# xtrain,xval,xtest = init_everything()
# print(xtrain.shape,xval.shape,xtest.shape)

xtrain = np.load("X_train_bandpass.npy")
xval = np.load("X_val_bandpass.npy")
xtest = np.load("X_test_bandpass.npy")
print(xtrain.shape)

# xtrain = np.load("X_train.npy")
# xval = np.load("X_val.npy")
# xtest = np.load("X_test.npy")

# ytrain = np.load("ytrain.npy")
# yval = np.load("yval.npy")
# ytest = np.load("ytest.npy")

# sletrain = np.load("sletrain.npy")
# sleval = np.load("sleval.npy")
# sletest = np.load("sletest.npy")


# np.save("X_train",xtrain)
# np.save("X_val",xval)
# np.save("X_test",xtest)

# ytrain,yval,ytest = P.get_data_y()
# sletrain,sleval,sletest = P.get_data_metadata()

# xtrain_bandpass = convert_bandpass(xtrain)
# xtest_bandpass = convert_bandpass(xtest)
# xval_bandpass = convert_bandpass(xval)

# xtrain_bandpass, ytrain, sletrain = segmentation(xtrain, ytrain, sletrain)
# xval_bandpass, yval, sleval = segmentation(xval, yval, sleval)
# xtest_bandpass, ytest, sletest = segmentation(xtest, ytest, sletest)

# print(xtrain_bandpass.shape)

# np.save("X_train_bandpass",xtrain_bandpass)
# np.save("X_val_bandpass",xval_bandpass)
# np.save("X_test_bandpass",xtest_bandpass)

# np.save("ytrain_t",ytrain), np.save("yval_t",yval), np.save("ytest_t",ytest)

# np.save("sletrain_t",sletrain), np.save("sleval_t",sleval), np.save("sletest_t",sletest)

# print("done")

# print("cwt-ing")
# cwt_data_scipy(xtrain,xval,xtest)
# stft_data(xtrain,xval,xtest)
# dwt_data(xtrain,xval,xtest)
# wvd_data(xtrain_bandpass,xval_bandpass,xtest_bandpass)
# fft_data(xtrain,xval,xtest)
