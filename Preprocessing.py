#taken and modified from https://github.com/helme/ecg_ptbxl_benchmarking
#truly a live saver
"""
File này chạy một lần là có đủ các cái data, labels, metadata save vào nhé
Tốc độ cao, và đặc biệt là nó chỉ ra là cái metric mình bị lmao
À với cả nó có cả mấy cái custom metric nữa kìa, ta có thể tham khảo
anh có thể đọc phần dưới, với việc không có thằng nào label full 0, chắc là sẽ đỡ noise hơn
"""
#default imports
import os
import sys
import re
import glob
import pickle 
import copy

#for loading and modifying data
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt #for plotting
from tqdm import tqdm #better progress bar
import wfdb
import ast

#for evaluation purposes (ở t cx skip phần này một chút)
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from matplotlib.axes._axes import _log as matplotlib_axes_logger
# import warnings

"""
First are the metrics used for calculating shits
We do not want to touch on those just yet.
"""
#ở đó họ có hết mấy hàm kia rồi, t đọc sốt cốt xong t thấy t ngu vcl, sorry a
def generate_results(idxs, y_true, y_pred, thresholds):
    return evaluate_experiment(y_true[idxs], y_pred[idxs], thresholds)

def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro') #DUCKKKKKKKKK
    
    df_result = pd.DataFrame(results, index=[0])
    return df_result

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1: 
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])): 
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0: 
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])): 
                FN += 1./sample_weight 
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    return {'F_beta_macro':f_beta/y_true.shape[1], 'G_beta_macro':g_beta/y_true.shape[1]}

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples

def find_optimal_cutoff_threshold(target, predicted):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def find_optimal_cutoff_thresholds(y_true, y_pred):
	return [find_optimal_cutoff_threshold(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]

def find_optimal_cutoff_threshold_for_Gbeta(target, predicted, n_thresholds=100):
    thresholds = np.linspace(0.00,1,n_thresholds)
    scores = [challenge_metrics(target, predicted>t, single=True)['G_beta_macro'] for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]

def find_optimal_cutoff_thresholds_for_Gbeta(y_true, y_pred):
    print("optimize thresholds with respect to G_beta")
    return [find_optimal_cutoff_threshold_for_Gbeta(y_true[:,k][:,np.newaxis], y_pred[:,k][:,np.newaxis]) for k in tqdm(range(y_true.shape[1]))]

def apply_thresholds(preds, thresholds):
	"""
		apply class-wise thresholds to prediction score in order to get binary format.
		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
	"""
	tmp = []
	for p in preds:
		tmp_p = (p > thresholds).astype(int)
		if np.sum(tmp_p) == 0:
			tmp_p[np.argmax(p)] = 1
		tmp.append(tmp_p)
	tmp = np.array(tmp)
	return tmp

#DATA PROCESSING STUFF
def load_dataset(path, sampling_rate, release=False):
    if True:
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        #Y ở đây thật ra là cái df, còn Y.scp_codes là để apply ra đống labels
        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
        #hàm load_raw_data sẽ được định nghĩa lại ở dưới

    return X, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    #df thật ra chính là cái data frame được load vào từ file csv
    if sampling_rate == 100:
        if False: #ở mấy chỗ này t để if false để thôi mình lưu một lần thôi đãm khi nào t rõ hơn vụ Pickle t sẽ xly nốt
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f,channels=[0,1,7]) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            # pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if False:
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f,channels=[0,1,7]) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    #ở đây họ trả cho ta raw data.
    return data

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0) #dòng này đọc ra các cái bệnh cần được classify

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
        meta = get_metadata(YY)
        meta = meta[YY.diagnostic_len > 0]
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
        meta = get_metadata(YY)
        meta = meta[YY.subdiagnostic_len > 0]
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
        meta = get_metadata(YY)
        meta = meta[YY.superdiagnostic_len > 0]
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
        meta = get_metadata(YY)
        meta = meta[YY.form_len > 0]
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
        meta = get_metadata(YY)
        meta = meta[YY.rhythm_len > 0]
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
        meta = get_metadata(YY)
        meta = meta[YY.all_scp_len > 0]
    else:
        pass

    # save LabelBinarizer
    # with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
    #     pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb, meta

def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    # with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
    #     pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def get_metadata(csv_file):
    #this is for computing the soft_label_encoding
    meta_sle = np.full((21799,32),0.1)
    meta_sle[csv_file.sex==1,0] = 1
    meta_sle[csv_file.sex==0,1] = 1
    for i in range(10):
        meta_sle[np.logical_and(np.array(csv_file.height)>=(100+10*i), np.array(csv_file.height)<(110+10*i)), 2+i] = 1 #height encoding
        meta_sle[np.logical_and(np.array(csv_file.weight)>=(0+12*i), np.array(csv_file.weight)<(12+12*i)), 12+i] = 1 #weight encoding
        meta_sle[np.logical_and(np.array(csv_file.age)>=(0+10*i), np.array(csv_file.age)<(10+10*i)), 22+i] = 1 #age encoding
    meta_sle[csv_file.age == 300, 31] = 1

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
    
    for i in range(21430):
            meta_sle[i] = fill_noise(meta_sle[i])
    return meta_sle

sampling_frequency=100
datafolder=''
task='superdiagnostic' #thay các thứ ở đây để nó ngon hơn
outputfolder='../output/' #không có cái outputfolder nào đâu, đống dùng đến nó t comment out hết rồi

# Load PTB-XL data
data, raw_labels = load_dataset(datafolder, sampling_frequency)
# Preprocess label data
labels = compute_label_aggregations(raw_labels, datafolder, task)
# Select relevant data and convert to one-hot
data, labels, Y, _, metadata = select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

# 1-9 for training 
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
metadata_train = metadata[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]
metadata_val = metadata[labels.strat_fold == 10]

num_classes = 5         # <=== number of classes in the finetuning dataset
input_shape = [1000,3] # <=== shape of samples, [None, 12] in case of different lengths


print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, metadata_train.shape, metadata_val.shape)
np.save("X_train",X_train)
np.save("X_val",X_val)

np.save("ytrain",y_train)
np.save("yval",y_val)

np.save("sletrain",metadata_train)
np.save("sleval",metadata_val)

#đống này đều vẫn phải chuyển sang tensor nhé
print("imdonesaving")
